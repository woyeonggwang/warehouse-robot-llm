using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

/// <summary>
/// 메뉴 Warehouse > Build N Areas : 창고 학습장(바닥·벽·선반 A1~D4·작업대 4곳·정차 지점·로봇)을 N 개 만든다.
/// 다시 실행하면 기존 TrainingAreas 를 지우고 새로 만든다. 배치 숫자는 WarehouseLayout 에 있음.
/// </summary>
public static class WarehouseBuilder
{
    const string MatDir = "Assets/02Resources/00Component/Materials";
    const float AreaSpacingX = 30f, AreaSpacingZ = 26f;

    [MenuItem("Warehouse/Build 1 Area")] static void Build1() => Build(1);
    [MenuItem("Warehouse/Build 4 Areas")] static void Build4() => Build(4);
    [MenuItem("Warehouse/Build 8 Areas")] static void Build8() => Build(8);

    public static void Build(int count)
    {
        EnsureTags(WarehouseLayout.ObstacleTags);
        var old = GameObject.Find("TrainingAreas");
        if (old != null) Undo.DestroyObjectImmediate(old);

        var root = new GameObject("TrainingAreas");
        Undo.RegisterCreatedObjectUndo(root, "Build Warehouse");
        for (int i = 0; i < count; i++)
        {
            var area = BuildArea($"Area{i}");
            area.transform.SetParent(root.transform, false);
            area.transform.localPosition = new Vector3((i % 4) * AreaSpacingX, 0f, (i / 4) * AreaSpacingZ);
        }
        EditorSceneManager.MarkSceneDirty(root.scene);
        Debug.Log($"[WarehouseBuilder] 학습장 {count}개 생성");
    }

    static GameObject BuildArea(string name)
    {
        var mFloor = Mat("Floor", new Color(0.55f, 0.55f, 0.58f));
        var mWall = Mat("Wall", new Color(0.82f, 0.82f, 0.85f));
        var mShelf = Mat("Shelf", new Color(0.25f, 0.42f, 0.75f));
        var mDock = Mat("DockPad", new Color(0.95f, 0.8f, 0.2f));
        var mTarget = Mat("Target", new Color(0.9f, 0.15f, 0.15f));
        var stationColors = new Dictionary<string, Color>
        {
            { "inbound", new Color(0.3f, 0.7f, 0.35f) }, { "outbound", new Color(0.85f, 0.45f, 0.2f) },
            { "packing", new Color(0.6f, 0.4f, 0.8f) },  { "charger", new Color(0.2f, 0.75f, 0.8f) },
        };

        var area = new GameObject(name);
        area.AddComponent<WarehouseArea>();
        float hx = WarehouseLayout.SizeX * 0.5f, hz = WarehouseLayout.SizeZ * 0.5f, wh = WarehouseLayout.WallHeight;

        // 바닥·벽
        var env = new GameObject("Environment").transform;
        env.SetParent(area.transform, false);
        Box("Floor", env, new Vector3(0f, -0.1f, 0f), new Vector3(WarehouseLayout.SizeX + 0.4f, 0.2f, WarehouseLayout.SizeZ + 0.4f), mFloor);
        Box("Wall_N", env, new Vector3(0f, wh * 0.5f, hz + 0.1f), new Vector3(WarehouseLayout.SizeX + 0.4f, wh, 0.2f), mWall, "Wall");
        Box("Wall_S", env, new Vector3(0f, wh * 0.5f, -hz - 0.1f), new Vector3(WarehouseLayout.SizeX + 0.4f, wh, 0.2f), mWall, "Wall");
        Box("Wall_E", env, new Vector3(hx + 0.1f, wh * 0.5f, 0f), new Vector3(0.2f, wh, WarehouseLayout.SizeZ), mWall, "Wall");
        Box("Wall_W", env, new Vector3(-hx - 0.1f, wh * 0.5f, 0f), new Vector3(0.2f, wh, WarehouseLayout.SizeZ), mWall, "Wall");

        var locs = new GameObject("Locations").transform;
        locs.SetParent(area.transform, false);

        // 선반 A~D × 칸 1~4
        var shelves = new GameObject("Shelves").transform;
        shelves.SetParent(env, false);
        for (int r = 0; r < WarehouseLayout.Rows.Length; r++)
            for (int b = 0; b < WarehouseLayout.BayX.Length; b++)
            {
                string id = WarehouseLayout.Rows[r] + (b + 1);
                var c = new Vector3(WarehouseLayout.BayX[b], WarehouseLayout.RackHeight * 0.5f, WarehouseLayout.RowZ[r]);
                Box("Shelf_" + id, shelves, c, new Vector3(WarehouseLayout.BayWidth - 0.1f, WarehouseLayout.RackHeight, WarehouseLayout.RackDepth), mShelf, "Shelf");
                Label(id, shelves, new Vector3(c.x, WarehouseLayout.RackHeight + 0.3f, c.z - WarehouseLayout.RackDepth * 0.5f), 0f);
                Location(id, locs, WarehouseLayout.ShelfDock(r, b), 0f, mDock);
            }

        // 작업대 (입고장·출고장·포장대·충전소)
        var stations = new GameObject("Stations").transform;
        stations.SetParent(env, false);
        foreach (var s in WarehouseLayout.Stations)
        {
            Box("Station_" + s.name, stations, s.blockCenter, s.blockSize, Mat("Station_" + s.name, stationColors[s.name]), "Station");
            Label(s.name, stations, s.blockCenter + Vector3.up * 0.9f, s.dockYaw);
            Location(s.name, locs, s.dock, s.dockYaw, mDock);
        }

        // 목표 표시 (충돌 없음)
        var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        marker.name = "TargetMarker";
        Object.DestroyImmediate(marker.GetComponent<Collider>());
        marker.transform.SetParent(area.transform, false);
        marker.transform.localScale = new Vector3(0.6f, 0.01f, 0.6f);
        marker.GetComponent<Renderer>().sharedMaterial = mTarget;

        BuildRobot(area.transform, marker.transform);
        return area;
    }

    static void BuildRobot(Transform area, Transform marker)
    {
        var mBody = Mat("RobotBody", new Color(0.95f, 0.6f, 0.1f));
        var mWheel = Mat("RobotWheel", new Color(0.12f, 0.12f, 0.12f));
        var mFront = Mat("RobotFront", new Color(0.1f, 0.1f, 0.1f));

        var robot = new GameObject("Robot");
        robot.transform.SetParent(area, false);
        robot.transform.localPosition = new Vector3(0f, 0.15f, -8.5f);

        var rb = robot.AddComponent<Rigidbody>();
        rb.mass = 30f;
        rb.linearDamping = 0.3f;
        rb.angularDamping = 2f;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        var body = Box("Body", robot.transform, new Vector3(0f, 0.12f, 0f), new Vector3(0.56f, 0.2f, 0.8f), mBody);
        body.isStatic = false;
        var front = Box("Front", robot.transform, new Vector3(0f, 0.25f, 0.3f), new Vector3(0.3f, 0.06f, 0.15f), mFront);
        front.isStatic = false;
        Object.DestroyImmediate(front.GetComponent<Collider>());

        var wl = Wheel("WheelL", robot.transform, -0.33f);
        var wr = Wheel("WheelR", robot.transform, 0.33f);
        var vl = WheelVisual("WheelL_Visual", robot.transform, -0.33f, mWheel);
        var vr = WheelVisual("WheelR_Visual", robot.transform, 0.33f, mWheel);

        // 앞뒤 보조 바퀴: 마찰 0 구체 (하중은 구동 바퀴가 받도록 살짝 띄움)
        var slick = SlickMaterial();
        foreach (float z in new[] { 0.33f, -0.33f })
        {
            var c = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            c.name = z > 0 ? "CasterF" : "CasterB";
            c.transform.SetParent(robot.transform, false);
            c.transform.localPosition = new Vector3(0f, -0.07f, z);
            c.transform.localScale = Vector3.one * 0.14f;
            c.GetComponent<Collider>().sharedMaterial = slick;
            c.GetComponent<Renderer>().sharedMaterial = mWheel;
        }

        // ML-Agents
        var agent = robot.AddComponent<RobotAgent>();
        agent.wheelL = wl; agent.wheelR = wr;
        agent.visualL = vl; agent.visualR = vr;
        agent.targetMarker = marker;
        agent.MaxStep = 3000;

        var bp = robot.GetComponent<BehaviorParameters>();
        if (bp == null) bp = robot.AddComponent<BehaviorParameters>();
        bp.BehaviorName = "GoTo";
        bp.BrainParameters.VectorObservationSize = 12;
        bp.BrainParameters.NumStackedVectorObservations = 1;
        bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);

        var dr = robot.AddComponent<DecisionRequester>();
        dr.DecisionPeriod = 5;
        dr.TakeActionsBetweenDecisions = true;

        var ray = robot.AddComponent<RayPerceptionSensorComponent3D>();
        ray.SensorName = "Rays";
        ray.DetectableTags = new List<string>(WarehouseLayout.ObstacleTags);
        ray.RaysPerDirection = 8;
        ray.MaxRayDegrees = 180f;
        ray.RayLength = 8f;
        ray.SphereCastRadius = 0.1f;
        ray.StartVerticalOffset = 0.2f;
        ray.EndVerticalOffset = 0.2f;
    }

    static WheelCollider Wheel(string name, Transform parent, float x)
    {
        var g = new GameObject(name);
        g.transform.SetParent(parent, false);
        g.transform.localPosition = new Vector3(x, 0f, 0f);
        var w = g.AddComponent<WheelCollider>();
        w.radius = 0.12f;
        w.mass = 1f;
        w.suspensionDistance = 0.05f;
        w.forceAppPointDistance = 0f;
        var s = w.suspensionSpring;
        s.spring = 15000f; s.damper = 1500f; s.targetPosition = 0.5f;
        w.suspensionSpring = s;
        return w;
    }

    static Transform WheelVisual(string name, Transform parent, float x, Material m)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        g.name = name;
        Object.DestroyImmediate(g.GetComponent<Collider>());
        g.transform.SetParent(parent, false);
        g.transform.localPosition = new Vector3(x, 0f, 0f);
        g.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
        g.transform.localScale = new Vector3(0.24f, 0.03f, 0.24f);
        g.GetComponent<Renderer>().sharedMaterial = m;
        return g.transform;
    }

    static void Location(string id, Transform parent, Vector3 pos, float yaw, Material pad)
    {
        var t = new GameObject(id).transform;
        t.SetParent(parent, false);
        t.localPosition = pos;
        t.localRotation = Quaternion.Euler(0f, yaw, 0f);
        var p = GameObject.CreatePrimitive(PrimitiveType.Cube);   // 바닥 정차 표시 (충돌 없음)
        p.name = "Pad";
        Object.DestroyImmediate(p.GetComponent<Collider>());
        p.transform.SetParent(t, false);
        p.transform.localPosition = new Vector3(0f, 0.005f, 0f);
        p.transform.localScale = new Vector3(0.8f, 0.01f, 0.8f);
        p.GetComponent<Renderer>().sharedMaterial = pad;
    }

    static void Label(string text, Transform parent, Vector3 pos, float yaw)
    {
        var g = new GameObject("Label_" + text);
        g.transform.SetParent(parent, false);
        g.transform.localPosition = pos;
        g.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);   // TextMesh 는 +z 방향을 바라보는 사람에게 읽힘
        var tm = g.AddComponent<TextMesh>();
        tm.text = text;
        tm.characterSize = 0.15f;
        tm.fontSize = 48;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.color = Color.white;
        tm.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        g.GetComponent<MeshRenderer>().sharedMaterial = tm.font.material;
    }

    static GameObject Box(string name, Transform parent, Vector3 localPos, Vector3 size, Material m, string tag = null)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(parent, false);
        g.transform.localPosition = localPos;
        g.transform.localScale = size;
        g.GetComponent<Renderer>().sharedMaterial = m;
        if (tag != null) g.tag = tag;
        g.isStatic = true;
        return g;
    }

    static Material Mat(string name, Color c)
    {
        if (!AssetDatabase.IsValidFolder(MatDir)) AssetDatabase.CreateFolder("Assets/02Resources/00Component", "Materials");
        string path = $"{MatDir}/{name}.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            AssetDatabase.CreateAsset(m, path);
        }
        m.SetColor("_BaseColor", c);
        EditorUtility.SetDirty(m);
        return m;
    }

    static PhysicsMaterial SlickMaterial()
    {
        if (!AssetDatabase.IsValidFolder(MatDir)) AssetDatabase.CreateFolder("Assets/02Resources/00Component", "Materials");
        string path = $"{MatDir}/Slick.physicMaterial";
        var m = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
        if (m == null)
        {
            m = new PhysicsMaterial("Slick");
            AssetDatabase.CreateAsset(m, path);
        }
        m.dynamicFriction = 0f;
        m.staticFriction = 0f;
        m.frictionCombine = PhysicsMaterialCombine.Minimum;
        EditorUtility.SetDirty(m);
        return m;
    }

    static void EnsureTags(IEnumerable<string> tags)
    {
        var tm = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        var p = tm.FindProperty("tags");
        foreach (var t in tags)
        {
            bool found = false;
            for (int i = 0; i < p.arraySize; i++) if (p.GetArrayElementAtIndex(i).stringValue == t) { found = true; break; }
            if (found) continue;
            p.InsertArrayElementAtIndex(p.arraySize);
            p.GetArrayElementAtIndex(p.arraySize - 1).stringValue = t;
        }
        tm.ApplyModifiedProperties();
    }
}
