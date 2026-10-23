using UnityEngine;
using Unity.XR.CoreUtils;
using UnityEngine.InputSystem;

public class XRCameraController : MonoBehaviour
{
    // Existing serialized fields
    [SerializeField] private XROrigin xrOrigin;
    [SerializeField] private Vector3 firstPersonOffset = new Vector3(0, 0, 0);
    [SerializeField] private Vector3 thirdPersonOffset = new Vector3(0f, 1.5f, -2f);
    [SerializeField] private float lookAheadDistance = 4f;
    [SerializeField] private float smoothSpeed = 10f;
    [SerializeField] private float rotationSmoothSpeed = 5f;
    [SerializeField] private float sideOffset = 3f;
    [SerializeField] private float gridWidth = 200f;
    [SerializeField] private float gridDepth = 200f;
    [SerializeField] private float viewHeight = 100f; // New default height for fly mode
    [SerializeField] private AgentRenderer agentRenderer;

    // New serialized fields for flying mode
    [Header("Flying Mode Settings")]
    [SerializeField] private float flySpeed = 5f;
    [SerializeField] private float flyRotationSpeed = 90f;
    [SerializeField] private InputActionReference flyMovementAction;
    [SerializeField] private InputActionReference flyRotationAction;

    // Existing and new InputActionReferences
    [Header("Control Actions")]
    [SerializeField] private InputActionReference cycleModeAction;
    [SerializeField] private InputActionReference cycleAgentForwardAction;
    [SerializeField] private InputActionReference cycleAgentBackwardAction;
    [SerializeField] private InputActionReference toggleGridAction;

    [SerializeField] private InputActionReference toggleHudAction; // Action to toggle the HUD

    [Header("UI References")]
    [SerializeField] private GameObject hudCanvas; // Reference to the main HUD Canvas GameObject

    private enum CameraMode { FirstPerson, ThirdPerson, Global }
    private CameraMode currentMode = CameraMode.Global;
    private GameObject currentAgent;
    private Transform headTransform;
    private Vector3 movementDirection = Vector3.zero;
    private Vector3 smoothedDirection = Vector3.forward;
    private Vector3 lastValidDirection = Vector3.forward;

    // New variables to store input values and track mode state
    private Vector2 flyMovementInput = Vector2.zero;
    private Vector2 flyRotationInput = Vector2.zero;
    private bool isGlobalModeInitialized = false;

    void OnEnable()
    {
        // Existing action subscriptions
        if (cycleModeAction != null)
        {
            cycleModeAction.action.performed += OnCycleMode;
        }
        if (cycleAgentForwardAction != null)
        {
            cycleAgentForwardAction.action.performed += OnCycleAgentForward;
        }
        if (cycleAgentBackwardAction != null)
        {
            cycleAgentBackwardAction.action.performed += OnCycleAgentBackward;
        }
        if (flyMovementAction != null)
        {
            flyMovementAction.action.performed += OnFlyMovement;
            flyMovementAction.action.canceled += OnFlyMovement;
        }
        if (flyRotationAction != null)
        {
            flyRotationAction.action.performed += OnFlyRotation;
            flyRotationAction.action.canceled += OnFlyRotation;
        }
        if (toggleHudAction != null)
        {
            toggleHudAction.action.performed += OnToggleHud;
        }
    }

    void OnDisable()
    {
        // Existing action unsubscriptions
        if (cycleModeAction != null)
        {
            cycleModeAction.action.performed -= OnCycleMode;
        }
        if (cycleAgentForwardAction != null)
        {
            cycleAgentForwardAction.action.performed -= OnCycleAgentForward;
        }
        if (cycleAgentBackwardAction != null)
        {
            cycleAgentBackwardAction.action.performed -= OnCycleAgentBackward;
        }
        if (flyMovementAction != null)
        {
            flyMovementAction.action.performed -= OnFlyMovement;
            flyMovementAction.action.canceled -= OnFlyMovement;
        }
        if (flyRotationAction != null)
        {
            flyRotationAction.action.performed -= OnFlyRotation;
            flyRotationAction.action.canceled -= OnFlyRotation;
        }
        // START OF NEW CODE
        if (toggleHudAction != null)
        {
            toggleHudAction.action.performed -= OnToggleHud;
        }
        // END OF NEW CODE
    }

    void Start()
    {
        if (xrOrigin == null)
        {
            xrOrigin = FindObjectOfType<XROrigin>();
            if (xrOrigin == null)
            {
                Debug.LogError("XR Origin not found in scene!");
            }
        }
        if (agentRenderer == null)
        {
            agentRenderer = FindObjectOfType<AgentRenderer>();
            if (agentRenderer == null)
            {
                Debug.LogError("AgentRenderer not found in scene!");
            }
        }
        Debug.Log("XRCameraController initialized. Waiting for agent selection.");
    }

    private void OnToggleHud(InputAction.CallbackContext context)
    {
        if (hudCanvas != null)
        {
            bool newState = !hudCanvas.gameObject.activeSelf;
            hudCanvas.gameObject.SetActive(newState);
            Debug.Log($"HUD Canvas toggled: {newState}");
        }
        else
        {
            Debug.LogWarning("HUD Canvas is not assigned, cannot toggle display.");
        }
    }

    // Existing input callbacks
    private void OnFlyMovement(InputAction.CallbackContext context)
    {
        flyMovementInput = context.ReadValue<Vector2>();
    }

    private void OnFlyRotation(InputAction.CallbackContext context)
    {
        flyRotationInput = context.ReadValue<Vector2>();
    }

    private void OnCycleMode(InputAction.CallbackContext context)
    {
        currentMode = (CameraMode)(((int)currentMode + 1) % 3);
        Debug.Log($"Camera mode: {currentMode}");
        isGlobalModeInitialized = false;
    }

    private void OnCycleAgentForward(InputAction.CallbackContext context)
    {
        if (agentRenderer != null)
        {
            agentRenderer.SelectNextAgent();
            Debug.Log("Cycling to next agent");
        }
        else
        {
            Debug.LogWarning("AgentRenderer is not assigned, cannot cycle agents");
        }
    }

    private void OnCycleAgentBackward(InputAction.CallbackContext context)
    {
        if (agentRenderer != null)
        {
            agentRenderer.SelectPreviousAgent();
            Debug.Log("Cycling to previous agent");
        }
        else
        {
            Debug.LogWarning("AgentRenderer is not assigned, cannot cycle agents");
        }
    }

    void Update()
    {
        if (currentMode == CameraMode.Global)
        {
            HandleGlobalMode();
        }
        else if (currentAgent == null)
        {
            Debug.LogWarning("No agent assigned to XR camera.");
            return;
        }
        else
        {
            Transform targetTransform = headTransform != null ? headTransform : currentAgent.transform;
            Vector3 targetPosition = targetTransform.position;

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
                xrOrigin.transform.position = Vector3.Lerp(xrOrigin.transform.position, targetPosition + firstPersonOffset, smoothSpeed * Time.deltaTime);
                Debug.Log($"First-person XR Origin at {xrOrigin.transform.position}");
            }
            else // ThirdPerson
            {
                Vector3 combinedOffset = thirdPersonOffset + (Quaternion.Euler(0, 90, 0) * thirdPersonOffset.normalized * sideOffset);
                Quaternion rotation = Quaternion.LookRotation(forwardDirection, Vector3.up);
                Vector3 rotatedOffset = rotation * combinedOffset;
                Vector3 desiredPosition = targetTransform.position + rotatedOffset;

                xrOrigin.transform.position = Vector3.Lerp(xrOrigin.transform.position, desiredPosition, smoothSpeed * Time.deltaTime);
                Debug.Log($"Third-person XR Origin at {xrOrigin.transform.position}");
            }
        }
    }

    private void HandleGlobalMode()
    {
        if (!isGlobalModeInitialized)
        {
            Vector3 currentPosition = xrOrigin.transform.position;
            xrOrigin.transform.position = new Vector3(currentPosition.x, viewHeight, currentPosition.z);
            isGlobalModeInitialized = true;
        }

        Vector3 cameraForward = xrOrigin.transform.forward;
        Vector3 cameraRight = xrOrigin.transform.right;
        cameraForward.y = 0;
        cameraForward.Normalize();
        cameraRight.y = 0;
        cameraRight.Normalize();

        Vector3 movement = cameraForward * flyMovementInput.y * flySpeed * Time.deltaTime;
        movement += cameraRight * flyMovementInput.x * flySpeed * Time.deltaTime;

        Vector3 newPosition = xrOrigin.transform.position + movement;
        xrOrigin.transform.position = new Vector3(newPosition.x, viewHeight, newPosition.z);

        float rotationAmount = flyRotationInput.x * flyRotationSpeed * Time.deltaTime;
        xrOrigin.transform.Rotate(Vector3.up, rotationAmount);
        Debug.Log($"Global mode at {xrOrigin.transform.position}");
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
        Debug.Log($"XR Camera following agent {agentId} at position {agentObject.transform.position}, direction={direction}");
    }
}
