using UnityEngine;

public class CameraController : MonoBehaviour
{
    [SerializeField] private Vector3 firstPersonOffset = new Vector3(0, 1.5f, 0);
    [SerializeField] private Vector3 thirdPersonOffset = new Vector3(0f, 3f, -2f);
    [SerializeField] private float lookAheadDistance = 4f;
    [SerializeField] private float smoothSpeed = 10f;
    [SerializeField] private float rotationSmoothSpeed = 5f;
    [SerializeField] private float sideOffset = 3f; // For third-person side offset
    [SerializeField] private float gridWidth = 200f; // Grid width (x-axis, Unity units)
    [SerializeField] private float gridDepth = 200f; // Grid depth (z-axis, Unity units)
    [SerializeField] private float viewHeight = 100f; // Camera height above grid (y-axis)

    private enum CameraMode { FirstPerson, ThirdPerson, Global }
    private CameraMode currentMode = CameraMode.ThirdPerson;
    private GameObject currentAgent;
    private Transform headTransform;
    private Vector3 movementDirection = Vector3.zero;
    private Vector3 smoothedDirection = Vector3.forward;
    private Vector3 lastValidDirection = Vector3.forward;

    void Start()
    {
        transform.SetParent(null);
        Debug.Log("CameraController initialized. Waiting for agent selection.");
    }

    void Update()
    {
        // Cycle through camera modes with 'C'
        if (Input.GetKeyDown(KeyCode.C))
        {
            currentMode = (CameraMode)(((int)currentMode + 1) % 3); // Cycle: 0, 1, 2
            Debug.Log($"Camera mode: {currentMode}");
        }

        if (currentMode == CameraMode.Global)
        {
            // Global view: Position above grid center, looking down
            Vector3 gridCenter = new Vector3(gridWidth / 2f, 0f, gridDepth / 2f);
            Vector3 desiredPosition = new Vector3(gridWidth / 2f, viewHeight, gridDepth / 2f);
            transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);

            Quaternion targetRotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSmoothSpeed * Time.deltaTime);

            Debug.Log($"Global camera at {transform.position}, looking at {gridCenter}");
        }
        else if (currentAgent == null)
        {
            Debug.LogWarning("No agent assigned to camera.");
            return;
        }
        else
        {
            // First-person or third-person
            Transform targetTransform = headTransform != null ? headTransform : currentAgent.transform;
            Vector3 targetPosition = headTransform != null ? targetTransform.position :
                currentAgent.transform.position + firstPersonOffset;

            if (movementDirection != Vector3.zero)
            {
                smoothedDirection = Vector3.Slerp(smoothedDirection, movementDirection, rotationSmoothSpeed * Time.deltaTime);
                lastValidDirection = movementDirection;
            }
            else
            {
                smoothedDirection = Vector3.Slerp(smoothedDirection, lastValidDirection, rotationSmoothSpeed * Time.deltaTime);
            }

            Vector3 forwardDirection = smoothedDirection.normalized;

            if (currentMode == CameraMode.FirstPerson)
            {
                transform.position = Vector3.Lerp(transform.position, targetPosition, smoothSpeed * Time.deltaTime);
                Quaternion targetRotation = Quaternion.LookRotation(forwardDirection, Vector3.up);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSmoothSpeed * Time.deltaTime);
                Debug.Log($"First-person camera at {transform.position}, direction={forwardDirection}");
            }
            else // ThirdPerson
            {
                Vector3 combinedOffset = thirdPersonOffset + (Quaternion.Euler(0, 90, 0) * thirdPersonOffset.normalized * sideOffset);
                Quaternion rotation = Quaternion.LookRotation(forwardDirection, Vector3.up);
                Vector3 rotatedOffset = rotation * combinedOffset;
                Vector3 desiredPosition = targetTransform.position + rotatedOffset;

                transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);

                Vector3 lookAtPoint = targetTransform.position + forwardDirection * lookAheadDistance;
                Vector3 directionToLookAt = (lookAtPoint - transform.position).normalized;
                Quaternion targetRotation = Quaternion.LookRotation(directionToLookAt, Vector3.up);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSmoothSpeed * Time.deltaTime);

                Debug.Log($"Third-person camera at {transform.position}, looking at {lookAtPoint}, direction={forwardDirection}");
            }
        }
    }

    public void SetAgent(int agentId, GameObject agentObject, Vector3 direction)
    {
        if (agentObject == null)
        {
            Debug.LogError($"SetAgent called with null agent object for agent {agentId}");
            return;
        }
        currentAgent = agentObject;
        movementDirection = direction;
        headTransform = agentObject.transform.Find("Head");
        if (headTransform == null)
        {
            Debug.LogWarning($"Agent {agentId} has no 'Head' child. Using firstPersonOffset {firstPersonOffset}.");
        }
        Debug.Log($"Camera following agent {agentId} at position {agentObject.transform.position}, direction={direction}");
    }
}