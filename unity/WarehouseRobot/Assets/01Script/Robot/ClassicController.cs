using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// 비교 기준(고전 방식): NavMesh 경로 계획 + Pure Pursuit 추종 + PID 로 바퀴 토크 계산.
/// 강화학습 정책과 같은 로봇·같은 바퀴 물리·같은 행동 처리(토크 -1~1, 속도·회전 제한)를 쓰고, 행동을 만드는 부분만 바꾼다.
/// RobotAgent.controller = Classic 이면 Heuristic() 에서 이 클래스가 행동을 만든다 (BehaviorType = HeuristicOnly).
/// </summary>
public class ClassicController
{
    // 조정값
    public float lookahead = 1.0f;      // 따라갈 점 거리 (m)
    public float cruiseSpeed = 1.4f;    // 목표 속도 (m/s), 로봇 최고 1.5
    public float speedGain = 2.0f;      // 속도 오차(m/s) → 전진 명령
    public float turnGain = 1.5f;       // 방향 오차(rad) → 회전 명령
    public float yawDamping = 0.3f;     // 회전 속도(rad/s) 감쇠
    public float spinAngle = 1.0f;      // 이 각도(rad, 약 57°) 이상 틀어지면 제자리 회전
    public bool stopForWorkers = false; // 정지 규칙: 앞에 작업자가 있으면 멈춤 (움직이는 장애물 비교용)
    public float stopBase = 0.8f;       // 정지 거리 = stopBase + 속도 × stopPerSpeed (m)
    public float stopPerSpeed = 0.6f;

    readonly List<Vector3> corners = new List<Vector3>();
    public float PathLength { get; private set; }
    public bool HasPath => corners.Count >= 2;

    public void Plan(Vector3 from, Vector3 to)
    {
        corners.Clear();
        PathLength = 0f;
        WarehouseNavMesh.Ensure();
        if (!NavMesh.SamplePosition(from, out var a, 1.5f, NavMesh.AllAreas)) return;
        if (!NavMesh.SamplePosition(to, out var b, 1.5f, NavMesh.AllAreas)) return;
        var path = new NavMeshPath();
        if (!NavMesh.CalculatePath(a.position, b.position, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) return;
        corners.AddRange(path.corners);
        for (int i = 1; i < corners.Count; i++) PathLength += Vector3.Distance(corners[i - 1], corners[i]);
    }

    /// 경로 위에서 로봇과 가장 가까운 점부터 lookahead 만큼 앞의 점
    Vector3 LookPoint(Vector3 p)
    {
        int best = 0; float bestD = float.MaxValue, bestT = 0f;
        for (int i = 0; i < corners.Count - 1; i++)
        {
            Vector3 a = corners[i], b = corners[i + 1];
            Vector3 ab = b - a; ab.y = 0f;
            float len2 = Mathf.Max(ab.sqrMagnitude, 1e-6f);
            Vector3 ap = p - a; ap.y = 0f;
            float t = Mathf.Clamp01(Vector3.Dot(ap, ab) / len2);
            Vector3 q = a + ab * t; q.y = p.y;
            float d = (q - p).sqrMagnitude;
            if (d < bestD) { bestD = d; best = i; bestT = t; }
        }
        float remain = lookahead;
        Vector3 cur = Vector3.Lerp(corners[best], corners[best + 1], bestT);
        for (int i = best; i < corners.Count - 1; i++)
        {
            Vector3 next = corners[i + 1];
            float seg = Vector3.Distance(cur, next);
            if (seg >= remain) return cur + (next - cur).normalized * remain;
            remain -= seg;
            cur = next;
        }
        return corners[corners.Count - 1];
    }

    bool WorkerAhead(Transform robot, float v)
    {
        float dist = stopBase + v * stopPerSpeed;
        var hits = Physics.BoxCastAll(robot.position + Vector3.up * 0.3f, new Vector3(0.45f, 0.25f, 0.1f), robot.forward, robot.rotation, dist + 0.45f, ~0, QueryTriggerInteraction.Ignore);
        foreach (var h in hits) if (h.collider.CompareTag("Worker")) return true;
        return false;
    }

    /// 좌·우 바퀴 명령 (-1~1)
    public void Act(Transform robot, Rigidbody rb, float maxSpeed, out float left, out float right)
    {
        left = right = 0f;
        if (!HasPath) return;
        Vector3 target = LookPoint(robot.position);
        Vector3 rel = robot.InverseTransformPoint(target);
        float alpha = Mathf.Atan2(rel.x, rel.z);                    // + 면 오른쪽
        float v = Vector3.Dot(rb.linearVelocity, robot.forward);
        float yawRate = rb.angularVelocity.y;

        float vDes = Mathf.Abs(alpha) > spinAngle ? 0f : cruiseSpeed * Mathf.Clamp01(Mathf.Cos(alpha));
        if (stopForWorkers && WorkerAhead(robot, Mathf.Max(v, 0f))) vDes = 0f;
        float fwd = Mathf.Clamp((vDes - v) * speedGain, -1f, 1f);
        float turn = Mathf.Clamp(turnGain * alpha - yawDamping * yawRate, -1f, 1f);
        left = Mathf.Clamp(fwd + turn, -1f, 1f);
        right = Mathf.Clamp(fwd - turn, -1f, 1f);
    }
}

/// <summary>
/// 학습장 전체(TrainingAreas)의 NavMesh 를 실행 중 한 번 만든다. 로봇 반지름 0.55m (경로 계획 격자의 inflate 와 같게).
/// 로봇 콜라이더는 빼고, 벽·선반·작업대·바닥 콜라이더로 만든다.
/// </summary>
public static class WarehouseNavMesh
{
    public const float AgentRadius = 0.55f;
    static NavMeshDataInstance instance;
    static bool built;

    public static void Ensure()
    {
        if (built && instance.valid) return;
        var root = GameObject.Find("TrainingAreas");
        if (root == null) return;

        var settings = NavMesh.GetSettingsByID(0);
        settings.agentRadius = AgentRadius;
        settings.agentHeight = 1f;
        settings.agentClimb = 0.05f;
        settings.agentSlope = 10f;

        var markups = new List<NavMeshBuildMarkup>();
        foreach (var agent in root.GetComponentsInChildren<RobotAgent>())
            markups.Add(new NavMeshBuildMarkup { root = agent.transform, ignoreFromBuild = true });
        foreach (var w in root.GetComponentsInChildren<WorkerWalker>())
            markups.Add(new NavMeshBuildMarkup { root = w.transform, ignoreFromBuild = true });

        var sources = new List<NavMeshBuildSource>();
        NavMeshBuilder.CollectSources(root.transform, ~0, NavMeshCollectGeometry.PhysicsColliders, 0, markups, sources);

        var bounds = new Bounds(Vector3.zero, Vector3.zero);
        bool first = true;
        foreach (var c in root.GetComponentsInChildren<Collider>())
        {
            if (first) { bounds = c.bounds; first = false; } else bounds.Encapsulate(c.bounds);
        }
        bounds.Expand(2f);

        var data = NavMeshBuilder.BuildNavMeshData(settings, sources, bounds, Vector3.zero, Quaternion.identity);
        instance = NavMesh.AddNavMeshData(data);
        built = true;
        Debug.Log($"[WarehouseNavMesh] NavMesh 생성 (소스 {sources.Count}개, 반지름 {AgentRadius}m)");
    }
}
