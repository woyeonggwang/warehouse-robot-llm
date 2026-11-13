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
    [Header("바퀴")]
    public WheelCollider wheelL;
    public WheelCollider wheelR;
    public Transform visualL;
    public Transform visualR;
    public float maxTorque = 6f;         // N·m (바퀴 하나)
    public float maxSpeed = 1.5f;        // m/s, 넘으면 같은 방향 토크를 끊음
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

    [Header("보상")]
    public float progressScale = 0.05f;        // 경로를 따라 1m 가까워질 때마다
    public float reachReward = 1f;
    public float collisionPenalty = 1f;        // 벽·선반·작업대에 닿으면 이만큼 빼고 종료
    public float abruptActionPenalty = 0.005f; // 급조작 벌점: 직전 결정 대비 행동 변화량² × 이 값
    public float timePenalty = 0.5f;           // 에피소드 전체에 나눠서

    [System.NonSerialized] Rigidbody rb;
    [System.NonSerialized] WarehouseArea area;
    [System.NonSerialized] Transform targetT;
    [System.NonSerialized] string target;
    [System.NonSerialized] float prevGeo, lastL, lastR;
    [System.NonSerialized] int outcome;      // 0 시간 초과, 1 도착, 2 충돌, 3 뒤집힘
    [System.NonSerialized] bool started;

    public string CurrentTarget => target;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        area = GetComponentInParent<WarehouseArea>();
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
        if (started)
        {
            var stats = Academy.Instance.StatsRecorder;
            stats.Add("GoTo/Success", outcome == 1 ? 1f : 0f);
            stats.Add("GoTo/Collision", outcome == 2 ? 1f : 0f);
            stats.Add("GoTo/Timeout", outcome == 0 ? 1f : 0f);
        }
        started = true;
        outcome = 0;

        target = string.IsNullOrEmpty(fixedTarget) ? area.RandomLocation() : fixedTarget;
        targetT = area.GetLocation(target);

        area.SampleSpawn(target, 2f, out var pos, out var rot);
        pos += Vector3.up * 0.15f;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.SetPositionAndRotation(pos, rot);
        rb.position = pos;
        rb.rotation = rot;
        wheelL.motorTorque = wheelR.motorTorque = 0f;
        wheelL.brakeTorque = wheelR.brakeTorque = 0f;

        if (targetMarker != null) targetMarker.position = targetT.position + Vector3.up * 0.01f;
        prevGeo = area.GeoDistance(target, transform.position);
        lastL = lastR = 0f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
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
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
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
        Drive(wheelL, fwdCmd + turnCmd, fwdSpeed);
        Drive(wheelR, fwdCmd - turnCmd, fwdSpeed);

        // 경로를 따라 가까워진 만큼
        float geo = area.GeoDistance(target, transform.position);
        AddReward(progressScale * (prevGeo - geo));
        prevGeo = geo;
        if (MaxStep > 0) AddReward(-timePenalty / MaxStep);

        Vector3 d = targetT.position - transform.position;
        d.y = 0f;
        if (d.magnitude < reachDistance)        // 도착 (속도 조건 없음, 2026-11-13 사용자 결정)
        {
            outcome = 1;
            AddReward(reachReward);
            EndEpisode();
            return;
        }
        if (Vector3.Dot(transform.up, Vector3.up) < 0.5f)   // 뒤집힘
        {
            outcome = 3;
            AddReward(-1f);
            EndEpisode();
        }
    }

    void Drive(WheelCollider w, float cmd, float fwdSpeed)
    {
        cmd = Mathf.Clamp(cmd, -1f, 1f);
        float torque = cmd * maxTorque;
        if (Mathf.Abs(fwdSpeed) > maxSpeed && Mathf.Sign(torque) == Mathf.Sign(fwdSpeed)) torque = 0f;
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
        foreach (var t in WarehouseLayout.ObstacleTags)
        {
            if (!c.collider.CompareTag(t)) continue;
            outcome = 2;                       // 충돌 → 종료 (2026-11-13 사용자 결정)
            AddReward(-collisionPenalty);
            EndEpisode();
            return;
        }
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
        var c = actionsOut.ContinuousActions;
        c[0] = Mathf.Clamp(v + h, -1f, 1f);
        c[1] = Mathf.Clamp(v - h, -1f, 1f);
    }
}
