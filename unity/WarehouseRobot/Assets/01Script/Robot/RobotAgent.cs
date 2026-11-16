using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// go_to 기술 (구조 B: 경로 계획 + 강화학습 제어)
/// - 경로 계획: WarehouseArea 의 최단 거리 격자가 목표까지의 경로를 정하고, 로봇 앞 1.5m·3m 지점을 '따라갈 점'으로 준다
/// - 강화학습: 따라갈 점을 보고 좌우 바퀴 토크(-1~1, 연속 2개)로 경로를 따라가며 벽·선반·작업대를 피한다
/// 관측(벡터 12개): 따라갈 점 2개(로봇 기준 x,z), 최종 목표(로봇 기준 x,z, 거리), 속도(x,z), 회전 속도, 직전 행동
///                 + 광선 센서. 창고 안 절대 위치는 넣지 않음 (지도를 외우지 않고 경로 따라가기를 배우게)
/// 보상: 경로를 따라 가까워진 만큼 +, 도착 +1(종료), 충돌 -1(종료), 급조작 벌점, 시간 -, 뒤집힘 -1(종료)
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class RobotAgent : Agent
{
    public enum ControllerType { RL, Classic }
    [Header("제어 방식")]
    [Tooltip("RL: 학습된 정책(또는 학습) / Classic: NavMesh 경로 + Pure Pursuit + PID (비교 기준, BehaviorType HeuristicOnly 로 사용)")]
    public ControllerType controller = ControllerType.RL;
    [Tooltip("평가용: 에피소드마다 NavMesh 최단 경로 길이를 재서 '이상적인 도착 시간' 계산에 씀")]
    public bool measurePath = false;

    [Header("바퀴")]
    public WheelCollider wheelL;
    public WheelCollider wheelR;
    public Transform visualL;
    public Transform visualR;
    public float maxTorque = 6f;         // N·m (바퀴 하나)
    public float maxSpeed = 1.5f;        // m/s, 넘으면 같은 방향 토크를 끊음
    public float maxReverseSpeed = 0.5f; // m/s, 후진은 이 속도까지만 (2026-11-13 사용자 결정: 실제 로봇처럼 뒤쪽은 느리게)
    public float maxYawRate = 1.5f;      // rad/s (약 86°/s), 넘으면 같은 방향 회전 토크를 끊음
    public float idleBrake = 1f;         // 명령이 거의 0 일 때 거는 브레이크
    public float sidewaysStiffness = 2f; // 바퀴 옆 미끄럼 마찰 (기본 1) — 직진 시 옆으로 쏠림 줄이기
    [Tooltip("직접 조종할 때 회전 입력 비율")] public float heuristicTurnScale = 0.5f;

    [Header("목표·경로")]
    public Transform targetMarker;
    [Tooltip("비우면 매 에피소드 무작위 위치. 위치 이름(A1, charger 등)을 넣으면 고정")]
    public string fixedTarget = "";
    public float reachDistance = 0.5f;
    public float lookNear = 1.5f;        // 따라갈 점 1 (m 앞)
    public float lookFar = 3f;           // 따라갈 점 2 (m 앞)

    [Header("시연 (LLM 연결)")]
    [Tooltip("켜면 에피소드 자동 재시작·무작위 목표·순간이동을 하지 않고, DemoSetGoal 로 받은 목표까지만 이동 (충돌해도 계속)")]
    public bool demoMode = false;
    [System.NonSerialized] public bool demoReached;
    [System.NonSerialized] public int demoCollisions;
    [System.NonSerialized] bool manualActive;
    [System.NonSerialized] float manualL, manualR;

    /// 시연: 목표 위치를 정하면 학습된 정책이 그 위치로 이동 (도착하면 demoReached = true, 정지)
    public void DemoSetGoal(string name)
    {
        target = name;
        targetT = area.GetLocation(name);
        prevGeo = area.GeoDistance(target, transform.position);
        demoReached = false;
        manualActive = false;
        if (targetMarker != null) targetMarker.position = targetT.position + Vector3.up * 0.01f;
    }

    public void DemoClearGoal() { target = null; targetT = null; }

    /// 시연: 정책 대신 직접 바퀴 명령 (스크립트 기술용, 안전 정지는 그대로 적용)
    public void DemoManual(bool on, float l = 0f, float r = 0f) { manualActive = on; manualL = l; manualR = r; }

    public WarehouseArea Area => area;
    public bool SafetyActive => safetyActive;

    [Header("안전 정지 (실제 로봇의 안전 스캐너 역할, 강화학습·고전 방식 공통)")]
    [Tooltip("앞쪽 보호 구역에 작업자가 들어오면 정책 출력과 상관없이 바퀴를 멈춤 (2026-11-13 사용자 결정)")]
    public bool safetyStop = true;
    public float safetyBase = 0.6f;          // 보호 구역 길이 = 기본 + 속도 × safetyPerSpeed (몸체 앞면부터, m)
    public float safetyPerSpeed = 0.6f;
    public float safetyHalfWidth = 0.45f;    // 보호 구역 반폭 (몸체 반폭 0.28 + 여유)
    public float safetyBrake = 30f;          // 정지 시 브레이크 토크

    [Header("보상")]
    public float progressScale = 0.05f;        // 경로를 따라 1m 가까워질 때마다
    public float reachReward = 1f;
    public float collisionPenalty = 1f;        // 벽·선반·작업대에 닿으면 이만큼 빼고 종료
    public float abruptActionPenalty = 0.005f; // 급조작 벌점: 직전 결정 대비 행동 변화량² × 이 값
    public float timePenalty = 0.5f;           // 에피소드 전체에 나눠서
    public float reversePenaltyPerMeter = 0.08f; // 후진 벌점: 후진한 거리 1m 마다 (경로 보상 0.05/m 보다 커서 후진으로 가까워지면 손해. 2026-11-13 사용자 결정)

    [System.NonSerialized] Rigidbody rb;
    [System.NonSerialized] WarehouseArea area;
    [System.NonSerialized] Transform targetT;
    [System.NonSerialized] string target;
    [System.NonSerialized] float prevGeo, lastL, lastR;
    [System.NonSerialized] int outcome;      // 0 시간 초과, 1 도착, 2 충돌, 3 뒤집힘
    [System.NonSerialized] bool started;

    [System.NonSerialized] int episodeSteps;
    [System.NonSerialized] Vector3 spawnLocal;
    [System.NonSerialized] float spawnGeo;
    [System.NonSerialized] ClassicController classic;
    [System.NonSerialized] bool safetyActive;
    [System.NonSerialized] int safetySteps, workerBumps;
    [System.NonSerialized] float pathLen;

    public string CurrentTarget => target;

    // 추론 모드 평가용 집계 (학습기 없이도 셈). 0 시간 초과, 1 도착, 2 충돌, 3 뒤집힘
    public static int[] EvalCounts = new int[4];
    public static long EvalSuccessSteps;     // 도착한 에피소드의 스텝 합 (물리 스텝)
    public static double EvalSuccessPathLen; // 도착한 에피소드의 NavMesh 최단 경로 길이 합 (measurePath 일 때)
    public static long EvalSteps, EvalSafetySteps;   // 전체 물리 스텝, 안전 정지 중인 스텝
    public static int EvalWorkerBumps;               // 멈춰 있는 로봇에 작업자가 닿은 횟수 (로봇 책임 아님, 충돌로 세지 않음)
    public static int EvalTimeoutInSafety;           // 시간 초과로 끝날 때 안전 정지 중이던 횟수 (작업자에게 막혀 갇힌 경우로 추정)
    public static int EvalTimeoutMostlySafety;       // 시간 초과 에피소드 중 절반 이상을 안전 정지로 보낸 횟수
    // 그룹별 집계 (여러 설정을 학습장별로 나눠 한 번에 비교할 때):
    // [0 시간초과, 1 도착, 2 충돌, 3 뒤집힘, 4 도착 스텝 합, 5 도착 경로 길이 합(cm), 6 시간초과 아닌 에피소드 스텝 합, 7 그중 안전 정지 스텝, 8 안전 정지 중 시간초과, 9 멈춘 로봇에 작업자 닿음]
    public static System.Collections.Generic.Dictionary<string, long[]> EvalByGroup = new System.Collections.Generic.Dictionary<string, long[]>();
    [System.NonSerialized] public string evalGroup = "";

    /// 고전 제어기 조정값을 바꾸고 그룹 이름을 붙임 (실행 중 호출)
    public void SetClassicParams(float cruise, float look, float spin, string group) => SetClassicParams(cruise, look, spin, false, group);

    public void SetClassicParams(float cruise, float look, float spin, bool stopForWorkers, string group)
    {
        classic = new ClassicController { cruiseSpeed = cruise, lookahead = look, spinAngle = spin, stopForWorkers = stopForWorkers };
        if (targetT != null) classic.Plan(transform.position, targetT.position);
        evalGroup = group;
    }
    public static System.Collections.Generic.Dictionary<string, int[]> EvalByTarget = new System.Collections.Generic.Dictionary<string, int[]>();
    public static System.Collections.Generic.List<string> CollisionLog = new System.Collections.Generic.List<string>();
    public static void ResetEval()
    {
        EvalCounts = new int[4]; EvalSuccessSteps = 0; EvalSuccessPathLen = 0;
        EvalSteps = 0; EvalSafetySteps = 0; EvalWorkerBumps = 0; EvalTimeoutInSafety = 0; EvalTimeoutMostlySafety = 0;
        EvalByTarget.Clear(); CollisionLog.Clear(); EvalByGroup.Clear();
    }

    /// 평가 결과를 CSV 로 저장 (저장소 rl/eval/): 충돌 한 건씩 + 목표 위치별 결과
    public static string DumpEval(string tag)
    {
        string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "../../../rl/eval"));
        System.IO.Directory.CreateDirectory(dir);
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("target,obstacle,tag,x,z,yaw,fwd_speed,side_speed,yaw_rate,contact_x,contact_z,steps,geo_to_target,euclid_to_target,spawn_x,spawn_z,spawn_geo");
        foreach (var line in CollisionLog) sb.AppendLine(line);
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, $"{tag}_collisions.csv"), sb.ToString(), new System.Text.UTF8Encoding(true));
        sb.Clear();
        sb.AppendLine("target,timeout,success,collision,flip");
        foreach (var kv in EvalByTarget) sb.AppendLine($"{kv.Key},{kv.Value[0]},{kv.Value[1]},{kv.Value[2]},{kv.Value[3]}");
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, $"{tag}_by_target.csv"), sb.ToString(), new System.Text.UTF8Encoding(true));
        return dir;
    }

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        area = GetComponentInParent<WarehouseArea>();
        // 작업자가 로봇을 돌아서 걷도록 NavMesh 장애물 표시 (경로를 깎지는 않고 작업자 지역 회피에만 쓰임)
        if (GetComponent<UnityEngine.AI.NavMeshObstacle>() == null)
        {
            var obs = gameObject.AddComponent<UnityEngine.AI.NavMeshObstacle>();
            obs.shape = UnityEngine.AI.NavMeshObstacleShape.Box;
            obs.center = new Vector3(0f, 0.25f, 0f);
            obs.size = new Vector3(0.7f, 0.6f, 0.95f);
            obs.carving = false;
        }
        rb.centerOfMass = new Vector3(0f, -0.05f, 0.08f);   // 살짝 앞쪽: 앞 보조바퀴에 하중을 줘 앞뒤 흔들림(→ 쏠림) 줄이기
        wheelL.ConfigureVehicleSubsteps(5f, 12, 15);          // 저속에서 바퀴 물리 안정화 (Unity 권장 방식)
        foreach (var w in new[] { wheelL, wheelR })
        {
            var sf = w.sidewaysFriction;
            sf.stiffness = sidewaysStiffness;
            w.sidewaysFriction = sf;
        }
    }

    public override void OnEpisodeBegin()
    {
        if (demoMode)
        {
            // 시연: 자동 재시작 없음 (목표는 DemoSetGoal 로)
            outcome = 0; episodeSteps = 0; safetySteps = 0; workerBumps = 0; safetyActive = false;
            lastL = lastR = 0f;
            return;
        }
        if (started)
        {
            var stats = Academy.Instance.StatsRecorder;
            stats.Add("GoTo/Success", outcome == 1 ? 1f : 0f);
            stats.Add("GoTo/Collision", outcome == 2 ? 1f : 0f);
            stats.Add("GoTo/Timeout", outcome == 0 ? 1f : 0f);
            stats.Add("GoTo/SafetyStopRatio", episodeSteps > 0 ? (float)safetySteps / episodeSteps : 0f);
            EvalSteps += episodeSteps; EvalSafetySteps += safetySteps; EvalWorkerBumps += workerBumps;
            if (outcome == 0 && safetyActive) EvalTimeoutInSafety++;
            if (outcome == 0 && safetySteps * 2 > episodeSteps) EvalTimeoutMostlySafety++;
            EvalCounts[outcome]++;
            if (outcome == 1) { EvalSuccessSteps += episodeSteps; EvalSuccessPathLen += pathLen; }
            if (!EvalByTarget.TryGetValue(target, out var bt)) EvalByTarget[target] = bt = new int[4];
            bt[outcome]++;
            if (!string.IsNullOrEmpty(evalGroup))
            {
                if (!EvalByGroup.TryGetValue(evalGroup, out var g)) EvalByGroup[evalGroup] = g = new long[10];
                g[outcome]++;
                if (outcome == 1) { g[4] += episodeSteps; g[5] += (long)(pathLen * 100f); }
                if (outcome != 0) { g[6] += episodeSteps; g[7] += safetySteps; }
                else if (safetyActive) g[8]++;
                g[9] += workerBumps;
            }
        }
        started = true;
        outcome = 0;
        episodeSteps = 0;
        safetySteps = 0;
        workerBumps = 0;
        safetyActive = false;

        target = string.IsNullOrEmpty(fixedTarget) ? area.RandomLocation() : fixedTarget;
        targetT = area.GetLocation(target);

        area.SampleSpawn(target, 2f, out var pos, out var rot);
        spawnLocal = area.transform.InverseTransformPoint(pos);
        spawnGeo = area.GeoDistance(target, pos);
        pos += Vector3.up * 0.15f;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.SetPositionAndRotation(pos, rot);
        rb.position = pos;
        rb.rotation = rot;
        wheelL.motorTorque = wheelR.motorTorque = 0f;
        wheelL.brakeTorque = wheelR.brakeTorque = 0f;

        if (targetMarker != null) targetMarker.position = targetT.position + Vector3.up * 0.01f;
        pathLen = 0f;
        if (controller == ControllerType.Classic || measurePath)
        {
            if (classic == null) classic = new ClassicController();   // 조정값은 SetClassicParams 로 바꾼 값 유지
            classic.Plan(pos, targetT.position);
            pathLen = classic.PathLength;
        }
        prevGeo = area.GeoDistance(target, transform.position);
        lastL = lastR = 0f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (targetT == null)   // 시연 중 목표 없음: 관측 개수(13)만 맞춰 둠 (행동은 쓰지 않음)
        {
            for (int i = 0; i < 12; i++) sensor.AddObservation(0f);
            sensor.AddObservation(safetyActive ? 1f : 0f);
            return;
        }
        Vector3 p1 = transform.InverseTransformPoint(area.PathPoint(target, transform.position, lookNear));
        Vector3 p2 = transform.InverseTransformPoint(area.PathPoint(target, transform.position, lookFar));
        Vector3 rel = transform.InverseTransformPoint(targetT.position);
        Vector3 v = transform.InverseTransformDirection(rb.linearVelocity);

        sensor.AddObservation(Mathf.Clamp(p1.x / lookNear, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(p1.z / lookNear, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(p2.x / lookFar, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(p2.z / lookFar, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.x / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.z / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp01(new Vector2(rel.x, rel.z).magnitude / 30f));
        sensor.AddObservation(v.x / maxSpeed);
        sensor.AddObservation(v.z / maxSpeed);
        sensor.AddObservation(rb.angularVelocity.y / 3f);
        sensor.AddObservation(lastL);
        sensor.AddObservation(lastR);
        sensor.AddObservation(safetyActive ? 1f : 0f);   // 안전 정지 중인지 (정책이 자기 출력이 무시되는 중인지 알 수 있게)
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (demoMode && (targetT == null || manualActive))
        {
            // 시연: 목표가 없으면 정지, 스크립트 기술이면 직접 명령 (안전 정지는 그대로)
            float vs = Vector3.Dot(rb.linearVelocity, transform.forward);
            safetyActive = safetyStop && WorkerInProtectiveField(Mathf.Max(vs, 0f));
            if (safetyActive || !manualActive)
            {
                wheelL.motorTorque = wheelR.motorTorque = 0f;
                wheelL.brakeTorque = wheelR.brakeTorque = safetyActive ? safetyBrake : idleBrake * 5f;
            }
            else
            {
                Drive(wheelL, manualL, vs);
                Drive(wheelR, manualR, vs);
            }
            return;
        }
        episodeSteps++;
        float l = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float r = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // 급조작 벌점 (결정과 결정 사이에는 같은 행동이 반복되므로 변화량 0)
        float dl = l - lastL, dr = r - lastR;
        AddReward(-abruptActionPenalty * (dl * dl + dr * dr));
        lastL = l;
        lastR = r;

        // 회전 속도 제한: 전진 성분은 두고 회전 성분만 끊음 (l>r 이면 오른쪽으로 돌고 angularVelocity.y > 0)
        float fwdCmd = (l + r) * 0.5f, turnCmd = (l - r) * 0.5f;
        float yaw = rb.angularVelocity.y;
        if (Mathf.Abs(yaw) > maxYawRate && yaw * turnCmd > 0f) turnCmd = 0f;
        float fwdSpeed = Vector3.Dot(rb.linearVelocity, transform.forward);
        safetyActive = safetyStop && WorkerInProtectiveField(Mathf.Max(fwdSpeed, 0f));
        if (safetyActive)
        {
            // 안전 정지: 정책 출력 무시, 바퀴 정지 (실제 로봇의 보호 구역 정지)
            safetySteps++;
            wheelL.motorTorque = wheelR.motorTorque = 0f;
            wheelL.brakeTorque = wheelR.brakeTorque = safetyBrake;
        }
        else
        {
            Drive(wheelL, fwdCmd + turnCmd, fwdSpeed);
            Drive(wheelR, fwdCmd - turnCmd, fwdSpeed);
        }

        // 후진 벌점 (방향 전환용 짧은 후진은 작게, 후진 주행은 손해)
        if (fwdSpeed < 0f) AddReward(-reversePenaltyPerMeter * (-fwdSpeed) * Time.fixedDeltaTime);

        // 경로를 따라 가까워진 만큼
        float geo = area.GeoDistance(target, transform.position);
        AddReward(progressScale * (prevGeo - geo));
        prevGeo = geo;
        if (MaxStep > 0) AddReward(-timePenalty / MaxStep);

        Vector3 d = targetT.position - transform.position;
        d.y = 0f;
        if (d.magnitude < reachDistance)        // 도착 (속도 조건 없음, 2026-11-13 사용자 결정)
        {
            if (demoMode) { demoReached = true; DemoClearGoal(); return; }
            outcome = 1;
            AddReward(reachReward);
            EndEpisode();
            return;
        }
        if (!demoMode && Vector3.Dot(transform.up, Vector3.up) < 0.5f)   // 뒤집힘
        {
            outcome = 3;
            AddReward(-1f);
            EndEpisode();
        }
    }

    bool WorkerInProtectiveField(float v)
    {
        float len = safetyBase + safetyPerSpeed * v;
        Vector3 center = transform.position + transform.forward * (0.4f + len * 0.5f) + Vector3.up * 0.4f;
        foreach (var c in Physics.OverlapBox(center, new Vector3(safetyHalfWidth, 0.4f, len * 0.5f), transform.rotation, ~0, QueryTriggerInteraction.Ignore))
            foreach (var t in WarehouseLayout.DynamicTags)
                if (c.CompareTag(t)) return true;
        return false;
    }

    void Drive(WheelCollider w, float cmd, float fwdSpeed)
    {
        cmd = Mathf.Clamp(cmd, -1f, 1f);
        float torque = cmd * maxTorque;
        if (fwdSpeed > maxSpeed && torque > 0f) torque = 0f;
        if (fwdSpeed < -maxReverseSpeed && torque < 0f) torque = 0f;
        w.motorTorque = torque;
        w.brakeTorque = Mathf.Abs(cmd) < 0.05f ? idleBrake : 0f;
    }

    void Update()
    {
        SyncVisual(wheelL, visualL);
        SyncVisual(wheelR, visualR);
    }

    static void SyncVisual(WheelCollider w, Transform vis)
    {
        if (w == null || vis == null) return;
        w.GetWorldPose(out var p, out var q);
        vis.SetPositionAndRotation(p, q * Quaternion.Euler(0f, 0f, 90f));   // 원기둥 축(Y)을 바퀴 축(X)으로
    }

    void OnCollisionEnter(Collision c)
    {
        if (!WarehouseLayout.IsAnyObstacle(c.collider)) return;   // 벽·선반·작업대·작업자
        if (c.collider.CompareTag("Worker") && rb.linearVelocity.magnitude < 0.1f)
        {
            workerBumps++;                 // 멈춰 있는 로봇에 작업자가 닿음 → 로봇 책임 아님, 충돌로 세지 않고 계속
            return;
        }
        if (demoMode) { demoCollisions++; Debug.LogWarning($"[Demo] 충돌: {c.collider.name}"); return; }
        LogCollision(c);
        outcome = 2;                       // 충돌 → 종료 (2026-11-13 사용자 결정)
        AddReward(-collisionPenalty);
        EndEpisode();
    }

    void LogCollision(Collision c)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        Transform a = area.transform;
        Vector3 lp = a.InverseTransformPoint(transform.position);
        float yawLocal = Vector3.SignedAngle(a.forward, transform.forward, a.up);
        Vector3 v = transform.InverseTransformDirection(rb.linearVelocity);
        Vector3 cp = c.contactCount > 0 ? transform.InverseTransformPoint(c.GetContact(0).point) : Vector3.zero;
        Vector3 d = targetT.position - transform.position; d.y = 0f;
        CollisionLog.Add(string.Format(inv, "{0},{1},{2},{3:F2},{4:F2},{5:F0},{6:F2},{7:F2},{8:F2},{9:F2},{10:F2},{11},{12:F2},{13:F2},{14:F2},{15:F2},{16:F2}",
            target, c.collider.name, c.collider.tag, lp.x, lp.z, yawLocal, v.z, v.x, rb.angularVelocity.y, cp.x, cp.z,
            episodeSteps, area.GeoDistance(target, transform.position), d.magnitude, spawnLocal.x, spawnLocal.z, spawnGeo));
    }

    void OnDrawGizmosSelected()
    {
        if (!Application.isPlaying || area == null || target == null) return;
        Gizmos.color = Color.green;
        Gizmos.DrawSphere(area.PathPoint(target, transform.position, lookNear), 0.15f);
        Gizmos.color = Color.cyan;
        Gizmos.DrawSphere(area.PathPoint(target, transform.position, lookFar), 0.15f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var c = actionsOut.ContinuousActions;
        if (controller == ControllerType.Classic)
        {
            if (classic == null) { classic = new ClassicController(); classic.Plan(transform.position, targetT.position); }
            classic.Act(transform, rb, maxSpeed, out float cl, out float cr);
            c[0] = cl;
            c[1] = cr;
            return;
        }
        // 직접 조종 시험용: W/S 앞뒤, A/D 회전
        float v = 0f, h = 0f;
#if ENABLE_INPUT_SYSTEM
        var k = Keyboard.current;
        if (k != null)
        {
            if (k.wKey.isPressed || k.upArrowKey.isPressed) v += 1f;
            if (k.sKey.isPressed || k.downArrowKey.isPressed) v -= 1f;
            if (k.dKey.isPressed || k.rightArrowKey.isPressed) h += 1f;
            if (k.aKey.isPressed || k.leftArrowKey.isPressed) h -= 1f;
        }
#endif
        h *= heuristicTurnScale;
        c[0] = Mathf.Clamp(v + h, -1f, 1f);
        c[1] = Mathf.Clamp(v - h, -1f, 1f);
    }
}
