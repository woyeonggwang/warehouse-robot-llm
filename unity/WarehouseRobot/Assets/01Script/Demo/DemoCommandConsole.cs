using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// 시연: 한국어 명령 → LLM 서버(llm/infer.py --serve) → 실행·확인·거부 + 기술 순서 → 로봇이 순서대로 실행.
/// - go_to: 강화학습 정책(GoTo 모델)이 목표까지 이동
/// - dock: 아직 학습 전이라 임시 스크립트 (정차 지점 근처로 go_to 후 정차 방향으로 제자리 회전)
/// - load / unload / charge / wait / stop: 스크립트 (짐 표시 생성·제거, 대기)
/// - follow: 아직 학습 전 → 건너뜀
/// 화면 왼쪽 위 창에서 명령을 입력하거나 예시 버튼을 누름. '확인' 응답이면 네/아니요 버튼.
/// 먼저 LLM 서버를 띄워야 함: (robot_llm 환경, llm 폴더) python infer.py --serve --port 8765
/// </summary>
public class DemoCommandConsole : MonoBehaviour
{
    public RobotAgent robot;
    public string serverUrl = "http://127.0.0.1:8765/command";
    public float goToTimeout = 120f;
    [Tooltip("명령 창 크기 기준 화면 높이 (화면 높이 ÷ 이 값 배율로 확대)")] public float referenceHeight = 720f;
    [Tooltip("글자 크기 (기준 화면 기준)")] public int fontSize = 16;

    [System.Serializable] class Req { public string text; }
    [System.Serializable] public class Step { public string skill; public string target; public string item; public float sec; }
    [System.Serializable] public class Resp
    {
        public string input; public string action; public Step[] plan; public string error;
        public float relevance; public float risk; public string reason; public string message; public int latency_ms;
    }

    static readonly string[] Examples =
    {
        "A3에서 상자 꺼내서 포장대에 갖다 놔",
        "충전소로 가",
        "B2 선반으로 이동해",
        "하역장으로 가",
        "출고장에 택배 내려놔",
        "거기서 10초 기다려",
    };

    string input = "A3에서 상자 꺼내서 포장대에 갖다 놔";
    Resp last;
    bool busy, awaitingConfirm, running;
    string status = "대기";
    readonly List<string> log = new List<string>();
    GameObject cargo;
    Coroutine runCo;
    Vector2 scroll;

    string imeComposing = "";

    void Start()
    {
#if ENABLE_LEGACY_INPUT_MANAGER
        // 한글 입력 (IMGUI 입력창은 예전 입력 관리자의 IME 설정을 따름 → Player Settings 의 Active Input Handling 을 Both 로 둬야 함)
        Input.imeCompositionMode = IMECompositionMode.On;
#endif
#if ENABLE_INPUT_SYSTEM
        // 한글 입력: 프로젝트가 Input System 전용이라 예전 Input.imeCompositionMode 를 못 씀 → Input System 으로 IME 켜기
        var kb = Keyboard.current;
        if (kb != null)
        {
            kb.SetIMEEnabled(true);
            kb.onIMECompositionChange += c => imeComposing = c.ToString();
        }
#endif
    }

    void Log(string s)
    {
        log.Add($"[{Time.time:F1}] {s}");
        if (log.Count > 40) log.RemoveAt(0);
        Debug.Log("[Demo] " + s);
    }

    // ---------------------------------------------------------------- LLM 서버 호출
    IEnumerator Send(string text)
    {
        busy = true; awaitingConfirm = false; last = null;
        status = "LLM 해석 중...";
        Log($"명령: {text}");
        var body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(new Req { text = text }));
        using (var req = new UnityWebRequest(serverUrl, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json; charset=utf-8");
            req.timeout = 60;
            yield return req.SendWebRequest();
            busy = false;
            if (req.result != UnityWebRequest.Result.Success)
            {
                status = "서버 연결 실패";
                Log($"서버 오류: {req.error} (infer.py --serve 실행 중인지 확인)");
                yield break;
            }
            last = JsonUtility.FromJson<Resp>(req.downloadHandler.text);
        }
        Log($"LLM: {last.action} — {last.message} (관계율 {last.relevance:F2}, 위험률 {last.risk:F2}, {last.latency_ms}ms)");
        if (last.action == "실행") StartPlan(last.plan);
        else if (last.action == "확인" && last.plan != null && last.plan.Length > 0) { awaitingConfirm = true; status = "확인 필요"; }
        else if (last.action == "확인") status = "다시 말씀해 주세요";   // 확인이지만 해석 모델 계획이 없음 (예: 없는 위치) → 실행할 것이 없어 네/아니요를 띄우지 않음
        else status = "거부됨";
    }

    void StartPlan(Step[] plan)
    {
        if (runCo != null) StopCoroutine(runCo);
        runCo = StartCoroutine(Run(plan));
    }

    // ---------------------------------------------------------------- 기술 실행
    IEnumerator Run(Step[] plan)
    {
        running = true;
        for (int i = 0; i < plan.Length; i++)
        {
            var s = plan[i];
            status = $"{i + 1}/{plan.Length} {Describe(s)}";
            Log($"실행 {status}");
            bool ok = true;
            switch (s.skill)
            {
                case "go_to": yield return GoTo(s.target, r => ok = r); break;
                case "dock": yield return Dock(s.target, r => ok = r); break;
                case "load": yield return new WaitForSeconds(1f); SetCargo(s.item); break;
                case "unload": yield return new WaitForSeconds(1f); DropCargo(); break;
                case "charge": status = "충전 중"; yield return new WaitForSeconds(2f); break;
                case "wait": yield return new WaitForSeconds(Mathf.Max(0f, s.sec)); break;
                case "stop": robot.DemoClearGoal(); robot.DemoManual(false); i = plan.Length; break;
                case "follow": Log("follow 는 아직 학습 전이라 건너뜀"); break;
                default: Log($"모르는 기술: {s.skill}"); break;
            }
            if (!ok) { status = $"실패: {Describe(s)}"; Log(status); running = false; yield break; }
        }
        robot.DemoClearGoal();
        robot.DemoManual(false);
        status = "완료";
        Log("계획 완료");
        running = false;
    }

    IEnumerator GoTo(string target, System.Action<bool> done)
    {
        if (string.IsNullOrEmpty(target) || System.Array.IndexOf(robot.Area.LocationNames, target) < 0)
        {
            Log($"모르는 위치: {target}");
            done(false);
            yield break;
        }
        robot.DemoSetGoal(target);
        float t0 = Time.time;
        while (!robot.demoReached && Time.time - t0 < goToTimeout) yield return null;
        done(robot.demoReached);
    }

    /// 임시 dock (학습 전): ① 정차 지점 0.6m 안으로 go_to ② 천천히 정확한 정차 지점(±0.1m)까지 접근
    /// ③ 정차 방향(선반·작업대를 바라봄)으로 제자리 회전. 정차 지점은 장애물에서 0.9m 떨어져 있어 회전해도 닿지 않음
    IEnumerator Dock(string target, System.Action<bool> done)
    {
        var loc = robot.Area.GetLocation(target);
        Vector3 d = loc.position - robot.transform.position; d.y = 0f;
        if (d.magnitude > 0.6f)
        {
            bool ok = false;
            yield return GoTo(target, r => ok = r);
            if (!ok) { done(false); yield break; }
        }
        // ② 정확한 위치로 천천히 (앞이면 전진, 뒤면 후진)
        float t0 = Time.time;
        while (Time.time - t0 < 8f)
        {
            Vector3 rel = robot.transform.InverseTransformPoint(loc.position);
            rel.y = 0f;
            if (rel.magnitude < 0.1f) break;
            bool back = rel.z < 0f;
            float ang = Mathf.Atan2(rel.x, Mathf.Abs(rel.z));          // 진행 방향 기준 좌우 오차 (rad)
            if (back) ang = -ang;
            float fwd = Mathf.Clamp(rel.magnitude * 0.8f, 0.15f, 0.35f) * (back ? -1f : 1f) * Mathf.Clamp01(Mathf.Cos(ang));
            float turn = Mathf.Clamp(ang * 0.8f, -0.4f, 0.4f);
            robot.DemoManual(true, fwd + turn, fwd - turn);
            yield return null;
        }
        // ③ 정차 방향으로 회전
        t0 = Time.time;
        while (Time.time - t0 < 8f)
        {
            float err = Vector3.SignedAngle(robot.transform.forward, loc.forward, Vector3.up);   // + 면 오른쪽으로 돌아야 함
            if (Mathf.Abs(err) < 4f) break;
            float turn = Mathf.Clamp(err / 60f, -0.6f, 0.6f);
            robot.DemoManual(true, turn, -turn);
            yield return null;
        }
        robot.DemoManual(false);
        yield return new WaitForSeconds(0.3f);
        Vector3 e = loc.position - robot.transform.position; e.y = 0f;
        Log($"정차: 위치 오차 {e.magnitude:F2}m, 방향 오차 {Mathf.Abs(Vector3.SignedAngle(robot.transform.forward, loc.forward, Vector3.up)):F0}°");
        done(true);
    }

    void SetCargo(string item)
    {
        DropCargo(false);
        cargo = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Destroy(cargo.GetComponent<Collider>());
        cargo.name = "Cargo_" + item;
        cargo.transform.SetParent(robot.transform, false);
        cargo.transform.localPosition = new Vector3(0f, 0.42f, -0.05f);
        cargo.transform.localScale = item == "tote" ? new Vector3(0.45f, 0.22f, 0.55f) : item == "parcel" ? new Vector3(0.35f, 0.18f, 0.35f) : new Vector3(0.45f, 0.35f, 0.45f);
        var m = cargo.GetComponent<Renderer>().material;
        m.SetColor("_BaseColor", item == "tote" ? new Color(0.2f, 0.4f, 0.9f) : item == "parcel" ? new Color(0.85f, 0.75f, 0.55f) : new Color(0.65f, 0.45f, 0.25f));
        Log($"싣기: {item}");
    }

    void DropCargo(bool log = true)
    {
        if (cargo == null) return;
        cargo.transform.SetParent(null, true);
        cargo.transform.position = robot.transform.position + robot.transform.forward * 0.9f + Vector3.up * 0.2f;
        Destroy(cargo, 3f);
        cargo = null;
        if (log) Log("내리기");
    }

    static string Describe(Step s)
    {
        switch (s.skill)
        {
            case "go_to": case "dock": return $"{s.skill}({s.target})";
            case "load": return $"load({s.item})";
            case "wait": return $"wait({s.sec:F0}초)";
            default: return s.skill;
        }
    }

    // ---------------------------------------------------------------- 화면
    void OnGUI()
    {
        // 화면 해상도에 맞춰 창 전체를 확대 (기준 720p). 고해상도 Game 화면에서 글씨가 너무 작던 문제
        float scale = Mathf.Max(1f, Screen.height / referenceHeight);
        GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(scale, scale, 1f));
        float vh = Screen.height / scale;
        GUI.skin.label.wordWrap = true;
        GUI.skin.label.fontSize = fontSize;
        GUI.skin.button.fontSize = fontSize;
        GUI.skin.textField.fontSize = fontSize;
        GUILayout.BeginArea(new Rect(10, 10, 480, vh - 20), GUI.skin.box);
        GUILayout.Label("<b>창고 로봇 — 한국어 명령</b>", new GUIStyle(GUI.skin.label) { richText = true, fontSize = fontSize + 3 });

        GUI.enabled = !busy;
        GUILayout.BeginHorizontal();
        input = GUILayout.TextField(input, GUILayout.Width(380), GUILayout.Height(fontSize + 12));
        if (GUILayout.Button("보내기") || (Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.Return && GUI.GetNameOfFocusedControl() == ""))
            if (!string.IsNullOrWhiteSpace(input)) StartCoroutine(Send(input.Trim()));
        GUILayout.EndHorizontal();
#if ENABLE_LEGACY_INPUT_MANAGER
        if (!string.IsNullOrEmpty(Input.compositionString)) imeComposing = Input.compositionString;
#endif
        if (!string.IsNullOrEmpty(imeComposing)) GUILayout.Label($"입력 중: {imeComposing}");
        GUILayout.Label("예시:");
        foreach (var e in Examples)
            if (GUILayout.Button(e)) { input = e; StartCoroutine(Send(e)); }
        GUI.enabled = true;

        GUILayout.Space(6);
        GUILayout.Label($"상태: {status}{(robot != null && robot.SafetyActive ? "  [안전 정지: 작업자]" : "")}");
        if (last != null)
        {
            GUILayout.Label($"판단: {last.action}  |  {last.message}");
            GUILayout.Label($"관계율 {last.relevance:F2} · 위험률 {last.risk:F2} · {last.latency_ms}ms");
            if (last.plan != null && last.plan.Length > 0)
            {
                var sb = new StringBuilder("계획: ");
                for (int i = 0; i < last.plan.Length; i++) sb.Append(i > 0 ? " → " : "").Append(Describe(last.plan[i]));
                GUILayout.Label(sb.ToString());
            }
            if (awaitingConfirm)
            {
                GUILayout.BeginHorizontal();
                if (GUILayout.Button("네 (실행)")) { awaitingConfirm = false; Log("사용자 확인: 네"); StartPlan(last.plan); }
                if (GUILayout.Button("아니요")) { awaitingConfirm = false; status = "취소"; Log("사용자 확인: 아니요"); }
                GUILayout.EndHorizontal();
            }
        }
        if (running && GUILayout.Button("중지"))
        {
            if (runCo != null) StopCoroutine(runCo);
            robot.DemoClearGoal(); robot.DemoManual(false);
            running = false; status = "중지됨"; Log("사용자 중지");
        }
        if (robot != null) GUILayout.Label($"충돌 {robot.demoCollisions}회");

        GUILayout.Space(6);
        scroll = GUILayout.BeginScrollView(scroll);
        for (int i = log.Count - 1; i >= 0; i--) GUILayout.Label(log[i]);
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }
}
