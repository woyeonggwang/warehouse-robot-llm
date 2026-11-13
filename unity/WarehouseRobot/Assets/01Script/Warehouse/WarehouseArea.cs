using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 창고 배치 상수. 에디터 빌더(WarehouseBuilder)와 실행 코드가 같이 쓴다.
/// 좌표는 학습장(Area) 기준 로컬 좌표, x 오른쪽 · z 앞쪽, 단위 m.
/// 위치 이름은 llm/schema.py 와 같다: 선반 A1~D4, inbound, outbound, packing, charger
/// </summary>
public static class WarehouseLayout
{
    public const float SizeX = 24f, SizeZ = 20f;          // 바닥 크기 (벽 안쪽)
    public const float WallHeight = 1.5f;
    public const float BayWidth = 2.4f, RackDepth = 1.0f, RackHeight = 2.0f;
    public const float DockGap = 0.9f;                    // 선반·작업대 앞면에서 정차 지점까지 거리

    public static readonly string[] Rows = { "A", "B", "C", "D" };
    public static readonly float[] RowZ = { 6f, 2f, -2f, -6f };        // 선반 줄 중심 z (앞면은 -z 쪽)
    public static readonly float[] BayX = { -3.6f, -1.2f, 1.2f, 3.6f }; // 칸 1~4 중심 x

    public struct Station
    {
        public string name;
        public Vector3 blockCenter, blockSize;
        public Vector3 dock;     // 정차 지점
        public float dockYaw;    // 정차 시 바라보는 방향 (도)
    }

    public static readonly Station[] Stations =
    {
        new Station { name = "inbound",  blockCenter = new Vector3(-11.3f, 0.5f,  6f), blockSize = new Vector3(1f, 1f, 2f), dock = new Vector3(-9.9f, 0f,  6f), dockYaw = -90f },
        new Station { name = "outbound", blockCenter = new Vector3( 11.3f, 0.5f,  6f), blockSize = new Vector3(1f, 1f, 2f), dock = new Vector3( 9.9f, 0f,  6f), dockYaw =  90f },
        new Station { name = "packing",  blockCenter = new Vector3( 11.3f, 0.5f, -6f), blockSize = new Vector3(1f, 1f, 2f), dock = new Vector3( 9.9f, 0f, -6f), dockYaw =  90f },
        new Station { name = "charger",  blockCenter = new Vector3(-11.3f, 0.5f, -6f), blockSize = new Vector3(1f, 1f, 2f), dock = new Vector3(-9.9f, 0f, -6f), dockYaw = -90f },
    };

    /// 선반 칸의 정차 지점 (선반 앞, 선반을 바라봄 → yaw 0)
    public static Vector3 ShelfDock(int row, int bay) =>
        new Vector3(BayX[bay], 0f, RowZ[row] - RackDepth * 0.5f - DockGap);

    public static readonly string[] ObstacleTags = { "Wall", "Shelf", "Station" };
}

/// <summary>
/// 학습장 하나. 장애물 격자와 위치별 최단 경로 거리(벽·선반을 돌아가는 거리)를 미리 계산해서
/// 보상(가까워진 만큼 +)과 시작 위치 뽑기에 쓴다. 직선 거리로 보상을 주면 선반 뒤에서 막히기 쉬워서.
/// </summary>
public class WarehouseArea : MonoBehaviour
{
    [Tooltip("격자 한 칸 크기 (m)")] public float cell = 0.25f;
    [Tooltip("로봇 크기만큼 장애물을 부풀리는 반경 (m)")] public float inflate = 0.4f;

    // 실행 중 계산 결과는 직렬화하지 않음 (Unity 가 private 배열·bool 을 플레이 진입 시 보존해서
    // 딕셔너리만 비고 built 는 true 로 남는 문제가 있었음)
    [System.NonSerialized] readonly Dictionary<string, Transform> locations = new Dictionary<string, Transform>();
    [System.NonSerialized] readonly Dictionary<string, float[]> dist = new Dictionary<string, float[]>();
    [System.NonSerialized] string[] names;
    [System.NonSerialized] bool[] free;
    [System.NonSerialized] int nx, nz;
    [System.NonSerialized] bool built;

    public string[] LocationNames { get { EnsureBuilt(); return names; } }

    public Transform GetLocation(string name) { EnsureBuilt(); return locations[name]; }

    public string RandomLocation() { EnsureBuilt(); return names[Random.Range(0, names.Length)]; }

    void EnsureBuilt()
    {
        if (built && dist.Count > 0) return;
        built = true;
        locations.Clear();
        dist.Clear();

        var root = transform.Find("Locations");
        foreach (Transform t in root) locations[t.name] = t;
        names = new List<string>(locations.Keys).ToArray();

        // 장애물 격자: 칸 중심에 로봇 크기 상자를 놓아 벽·선반·작업대와 겹치는지 확인
        nx = Mathf.CeilToInt(WarehouseLayout.SizeX / cell);
        nz = Mathf.CeilToInt(WarehouseLayout.SizeZ / cell);
        free = new bool[nx * nz];
        var half = new Vector3(inflate, 0.4f, inflate);
        for (int z = 0; z < nz; z++)
            for (int x = 0; x < nx; x++)
            {
                Vector3 c = transform.TransformPoint(CellCenter(x, z) + Vector3.up * 0.5f);
                bool blocked = false;
                foreach (var col in Physics.OverlapBox(c, half, transform.rotation, ~0, QueryTriggerInteraction.Ignore))
                    if (IsObstacle(col)) { blocked = true; break; }
                free[z * nx + x] = !blocked;
            }

        foreach (var kv in locations) dist[kv.Key] = Dijkstra(kv.Value.position);
    }

    static bool IsObstacle(Collider c)
    {
        foreach (var t in WarehouseLayout.ObstacleTags) if (c.CompareTag(t)) return true;
        return false;
    }

    Vector3 CellCenter(int x, int z) =>
        new Vector3(-WarehouseLayout.SizeX * 0.5f + (x + 0.5f) * cell, 0f, -WarehouseLayout.SizeZ * 0.5f + (z + 0.5f) * cell);

    bool ToCell(Vector3 world, out int x, out int z)
    {
        Vector3 l = transform.InverseTransformPoint(world);
        x = Mathf.FloorToInt((l.x + WarehouseLayout.SizeX * 0.5f) / cell);
        z = Mathf.FloorToInt((l.z + WarehouseLayout.SizeZ * 0.5f) / cell);
        return x >= 0 && x < nx && z >= 0 && z < nz;
    }

    float[] Dijkstra(Vector3 goalWorld)
    {
        var d = new float[nx * nz];
        for (int i = 0; i < d.Length; i++) d[i] = float.PositiveInfinity;
        if (!ToCell(goalWorld, out int gx, out int gz)) return d;
        int s = gz * nx + gx;
        d[s] = 0f;
        var pq = new SortedSet<(float, int)> { (0f, s) };
        while (pq.Count > 0)
        {
            var (cd, u) = pq.Min;
            pq.Remove(pq.Min);
            if (cd > d[u]) continue;
            int ux = u % nx, uz = u / nx;
            for (int dz = -1; dz <= 1; dz++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dz == 0) continue;
                    int vx = ux + dx, vz = uz + dz;
                    if (vx < 0 || vx >= nx || vz < 0 || vz >= nz) continue;
                    int v = vz * nx + vx;
                    if (!free[v]) continue;
                    bool diag = dx != 0 && dz != 0;
                    if (diag && (!free[uz * nx + vx] || !free[vz * nx + ux])) continue;  // 모서리 끼어 지나가기 금지
                    float nd = cd + (diag ? 1.41421f : 1f) * cell;
                    if (nd < d[v]) { d[v] = nd; pq.Add((nd, v)); }
                }
        }
        return d;
    }

    /// 목표까지 장애물을 돌아가는 최단 거리 (m). 로봇이 벽에 붙어 칸이 막혀 있으면 가까운 빈 칸 기준
    public float GeoDistance(string name, Vector3 world)
    {
        EnsureBuilt();
        var d = dist[name];
        if (ToCell(world, out int x, out int z))
        {
            for (int r = 0; r <= 4; r++)
            {
                float best = float.PositiveInfinity;
                for (int dz = -r; dz <= r; dz++)
                    for (int dx = -r; dx <= r; dx++)
                    {
                        if (Mathf.Max(Mathf.Abs(dx), Mathf.Abs(dz)) != r) continue;
                        int cx = x + dx, cz = z + dz;
                        if (cx < 0 || cx >= nx || cz < 0 || cz >= nz) continue;
                        float v = d[cz * nx + cx];
                        if (v < best) best = v;
                    }
                if (!float.IsInfinity(best)) return best + r * cell;
            }
        }
        return Vector3.Distance(world, locations[name].position) * 1.5f + 5f;
    }

    /// 경로 계획: 현재 위치에서 목표까지 최단 경로를 따라 ahead(m) 앞의 지점 (월드 좌표).
    /// 거리 격자에서 가장 가까워지는 이웃 칸으로 계속 내려가는 방식. 목표가 더 가까우면 목표 지점
    public Vector3 PathPoint(string name, Vector3 world, float ahead)
    {
        EnsureBuilt();
        var d = dist[name];
        Vector3 goal = locations[name].position;
        if (!ToCell(world, out int x, out int z)) return goal;
        // 시작 칸이 막혀 있으면 가까운 빈 칸에서 출발
        if (float.IsInfinity(d[z * nx + x]))
        {
            bool found = false;
            for (int r = 1; r <= 4 && !found; r++)
                for (int dz = -r; dz <= r && !found; dz++)
                    for (int dx = -r; dx <= r && !found; dx++)
                    {
                        int cx = x + dx, cz = z + dz;
                        if (cx < 0 || cx >= nx || cz < 0 || cz >= nz) continue;
                        if (!float.IsInfinity(d[cz * nx + cx])) { x = cx; z = cz; found = true; }
                    }
            if (!found) return goal;
        }
        float walked = 0f;
        for (int step = 0; step < 400 && walked < ahead; step++)
        {
            int cur = z * nx + x;
            if (d[cur] <= 0f) return goal;
            int bx = x, bz = z;
            float best = d[cur];
            for (int dz = -1; dz <= 1; dz++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dz == 0) continue;
                    int vx = x + dx, vz = z + dz;
                    if (vx < 0 || vx >= nx || vz < 0 || vz >= nz) continue;
                    if (dx != 0 && dz != 0 && (!free[z * nx + vx] || !free[vz * nx + x])) continue;
                    float v = d[vz * nx + vx];
                    if (v < best) { best = v; bx = vx; bz = vz; }
                }
            if (bx == x && bz == z) break;
            walked += (bx != x && bz != z ? 1.41421f : 1f) * cell;
            x = bx; z = bz;
        }
        if (d[z * nx + x] <= 0f) return goal;
        return transform.TransformPoint(CellCenter(x, z));
    }

    /// 목표에서 최단 거리 minGeo 이상 떨어진 빈 칸을 무작위로 골라 시작 위치로 (로컬 좌표·yaw 반환)
    public void SampleSpawn(string target, float minGeo, out Vector3 worldPos, out Quaternion worldRot)
    {
        EnsureBuilt();
        var d = dist[target];
        for (int tries = 0; tries < 500; tries++)
        {
            int i = Random.Range(0, d.Length);
            if (!free[i] || float.IsInfinity(d[i]) || d[i] < minGeo) continue;
            Vector3 local = CellCenter(i % nx, i / nx);
            worldPos = transform.TransformPoint(local);
            worldRot = transform.rotation * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            return;
        }
        worldPos = transform.TransformPoint(new Vector3(0f, 0f, -8.5f));
        worldRot = transform.rotation;
    }
}
