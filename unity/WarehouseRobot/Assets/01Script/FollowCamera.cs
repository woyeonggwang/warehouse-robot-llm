using UnityEngine;

/// <summary>
/// 확인용 카메라: 대상(로봇) 뒤쪽 위에서 따라간다. 학습에는 영향 없음.
/// target 이 비어 있으면 이름이 "Robot" 인 첫 오브젝트를 찾는다.
/// </summary>
public class FollowCamera : MonoBehaviour
{
    public Transform target;
    public float distance = 3.5f;      // 뒤로
    public float height = 3.5f;        // 위로
    public float lookAhead = 1f;     // 로봇 앞쪽을 봄
    public float smooth = 5f;

    void LateUpdate()
    {
        if (target == null)
        {
            var go = GameObject.Find("Robot");
            if (go == null) return;
            target = go.transform;
        }
        Vector3 fwd = target.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
        fwd.Normalize();

        Vector3 want = target.position - fwd * distance + Vector3.up * height;
        // 카메라와 로봇 사이에 벽이 있으면 벽 앞으로 당김 (벽 가까이에서 시작하면 벽만 보이던 문제)
        Vector3 from = target.position + Vector3.up * 0.5f;
        if (Physics.Linecast(from, want, out var hit) && hit.collider.CompareTag("Wall"))
            want = hit.point + (from - want).normalized * 0.3f;
        float k = 1f - Mathf.Exp(-smooth * Time.unscaledDeltaTime);
        transform.position = Vector3.Lerp(transform.position, want, k);
        transform.LookAt(target.position + fwd * lookAhead);
    }
}