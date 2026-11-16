using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// 움직이는 장애물: 창고 작업자. 학습장 안의 빈 위치를 무작위로 골라 NavMesh 로 걸어가고, 도착하면 잠깐 서 있다가 다음 위치로.
/// - 속도 0.8~1.2 m/s, 반지름 0.3m, 키 1.7m (태그 Worker, 운동학 Rigidbody + Capsule 콜라이더 → 로봇 충돌 판정에 걸림)
/// - 로봇을 돌아서 걸어감: NavMeshAgent 지역 회피가 로봇의 NavMeshObstacle 을 피함 (2026-11-13 사용자 결정: 실제 현장처럼 로봇은 멈추고 사람이 피함)
/// - yieldToRobot(기본 꺼짐): 켜면 앞쪽 60° 안 1.2m 이내 로봇 앞에서 멈춤 (이전 방식, 로봇도 멈추면 교착이 생겼음)
/// - 경로 계획(최단 거리 격자·NavMesh)에는 작업자가 없음 → 피하는 건 로봇(강화학습 정책 또는 고전 정지 규칙)의 몫
/// WarehouseArea.workerCount 만큼 실행 중에 만들어진다.
/// </summary>
[RequireComponent(typeof(NavMeshAgent))]
public class WorkerWalker : MonoBehaviour
{
    public WarehouseArea area;
    public Vector2 speedRange = new Vector2(0.8f, 1.2f);
    public Vector2 waitRange = new Vector2(0f, 2f);
    public bool yieldToRobot = false;
    public float yieldDistance = 1.2f;
    [Tooltip("걷는 중에 이 시간(초) 동안 거의 못 움직이면 목적지를 바꿈 (로봇·벽 사이에 막혀 로봇을 가두던 문제, 2026-11-13)")]
    public float stuckTime = 2f;
    float stuckFor;

    NavMeshAgent nav;
    float waitUntil = -1f;
    float speed;

    void Start()
    {
        nav = GetComponent<NavMeshAgent>();
        speed = Random.Range(speedRange.x, speedRange.y);
        nav.speed = speed;
        nav.angularSpeed = 360f;
        nav.acceleration = 4f;
        nav.radius = 0.3f;
        nav.height = 1.7f;
        nav.obstacleAvoidanceType = ObstacleAvoidanceType.MedQualityObstacleAvoidance;   // 로봇(NavMeshObstacle)·다른 작업자를 피해 걸음
        nav.avoidancePriority = Random.Range(40, 60);
        nav.autoBraking = true;
        PickNext();
    }

    void Update()
    {
        if (nav == null || !nav.isOnNavMesh) return;

        if (yieldToRobot) nav.isStopped = RobotAhead();

        if (!nav.pathPending && nav.remainingDistance < 0.3f)
        {
            stuckFor = 0f;
            if (waitUntil < 0f) waitUntil = Time.time + Random.Range(waitRange.x, waitRange.y);
            else if (Time.time >= waitUntil) { waitUntil = -1f; PickNext(); }
        }
        else if (!nav.pathPending && !nav.isStopped)
        {
            // 걸어가야 하는데 막혀 있으면 (사람이라면) 다른 길로 감
            stuckFor = nav.velocity.magnitude < 0.15f ? stuckFor + Time.deltaTime : 0f;
            if (stuckFor > stuckTime) { stuckFor = 0f; waitUntil = -1f; PickNext(); }
        }
    }

    bool RobotAhead()
    {
        foreach (var c in Physics.OverlapSphere(transform.position + Vector3.up * 0.3f, yieldDistance, ~0, QueryTriggerInteraction.Ignore))
        {
            var agent = c.GetComponentInParent<RobotAgent>();
            if (agent == null) continue;
            Vector3 d = agent.transform.position - transform.position;
            d.y = 0f;
            Vector3 v = nav.velocity.sqrMagnitude > 0.01f ? nav.velocity : transform.forward;
            v.y = 0f;
            if (Vector3.Angle(v, d) < 60f) return true;
        }
        return false;
    }

    void PickNext()
    {
        if (area != null && nav.isOnNavMesh) nav.SetDestination(area.RandomFreePoint());
    }

    static Material mat;

    /// 작업자 하나를 만들어 학습장에 둔다
    public static WorkerWalker Create(WarehouseArea area, Vector3 pos, int index)
    {
        var go = new GameObject($"Worker{index}");
        go.transform.SetParent(area.transform, true);
        go.transform.position = pos;
        go.tag = "Worker";
        var col = go.AddComponent<CapsuleCollider>();
        col.center = new Vector3(0f, 0.85f, 0f);
        col.radius = 0.3f;
        col.height = 1.7f;
        var rb = go.AddComponent<Rigidbody>();
        rb.isKinematic = true;

        var vis = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        Object.Destroy(vis.GetComponent<Collider>());
        vis.name = "Body";
        vis.transform.SetParent(go.transform, false);
        vis.transform.localPosition = new Vector3(0f, 0.85f, 0f);
        vis.transform.localScale = new Vector3(0.6f, 0.85f, 0.6f);
        if (mat == null)
        {
            mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            mat.SetColor("_BaseColor", new Color(0.95f, 0.85f, 0.1f));   // 안전 조끼 색
        }
        vis.GetComponent<Renderer>().sharedMaterial = mat;

        var nav = go.AddComponent<NavMeshAgent>();
        nav.baseOffset = 0f;
        nav.Warp(pos);
        var w = go.AddComponent<WorkerWalker>();
        w.area = area;
        return w;
    }
}
