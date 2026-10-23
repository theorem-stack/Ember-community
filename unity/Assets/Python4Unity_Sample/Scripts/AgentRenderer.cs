using UnityEngine;
using System.Collections.Generic;
using System;
using System.Linq;

public class AgentRenderer : MonoBehaviour
{
    public GameObject agentPrefab; // Prefab for agent visualization
    public Material lineRendererMaterial; // Material for path LineRenderer
    [SerializeField] private float verticalOffset = 0.05f; // Offset to avoid terrain clipping for agents
    [SerializeField] private float pathHeightOffset = 0.5f; // Height above terrain for path rendering
    [SerializeField] private float stateUpdateInterval = 20f; // Time between state updates (20 seconds)
    [SerializeField] private int segmentsPerPoint = 10; // Number of spline segments per path point
    [SerializeField] private float raycastHeight = 100f; // Height above terrain to start raycast
    [SerializeField] private LayerMask terrainLayer; // Layer mask for terrain collider
    [SerializeField] private Material outlineMaterial; // Material for outline quad
    [SerializeField] private Material waterMaterial;
    [SerializeField] private Material firebreakMaterial;
    [SerializeField] private Material retardantMaterial;
    [SerializeField] private Material airSupportMaterial;
    [SerializeField] private Material idleMaterial;
    [SerializeField] private Material unknownMaterial;
    private InitialEnvironmentData initialData; // Store initial data for grid metadata
    private Dictionary<int, GameObject> agentObjects = new Dictionary<int, GameObject>(); // Track agents
    private Dictionary<int, LineRenderer> agentPaths = new Dictionary<int, LineRenderer>(); // Track paths
    private Dictionary<int, GameObject> agentTargets = new Dictionary<int, GameObject>(); // Track target markers
    private Dictionary<int, AgentState> agentStates = new Dictionary<int, AgentState>(); // Track interpolation states
    private int selectedAgentId = -1; // Currently selected agent (-1 means none)
    private List<int> agentIds = new List<int>(); // List of active agent IDs
    private Dictionary<int, Vector3> previousPositions = new Dictionary<int, Vector3>(); // Track previous positions for movement direction

    // Event to notify camera of selected agent change and movement direction
    public event Action<int, GameObject, Vector3> OnAgentSelected;

    // Structure to store interpolation state for each agent
    private class AgentState
    {
        public Vector3 CurrentPosition; // Current position along path
        public Vector3 NextPosition; // Next state position from simulation
        public float InterpolationTime; // Time since last waypoint transition
        public AgentData CurrentAgentData; // Current agent data for path and target
        public bool IsInterpolating; // Whether interpolation is active
        public List<Vector3> InterpolationPoints; // Pre-calculated spline points for the path
        public int CurrentWaypointIndex; // Current high-level waypoint index
        public List<int> WaypointIndices; // Indices in InterpolationPoints where waypoints start
    }

    void Update()
    {
        // Move agents along pre-calculated interpolation points
        foreach (var kvp in agentStates)
        {
            int agentId = kvp.Key;
            AgentState state = kvp.Value;

            // Ensure all necessary data is available for interpolation.
            if (!state.IsInterpolating || state.InterpolationPoints == null || state.WaypointIndices.Count < 2) continue;

            // Increment the interpolation time for the current segment.
            state.InterpolationTime += Time.deltaTime;

            // Calculate the interpolation progress (t) for the current waypoint segment.
            float t = Mathf.Clamp01(state.InterpolationTime / stateUpdateInterval);

            // Determine the current waypoint segment.
            int currentWaypointIndex = state.CurrentWaypointIndex;
            int startPointIndex = state.WaypointIndices[currentWaypointIndex];

            // Find the end point of the current segment.
            int nextWaypointIndex = currentWaypointIndex + 1;
            int endPointIndex = nextWaypointIndex < state.WaypointIndices.Count
                                ? state.WaypointIndices[nextWaypointIndex]
                                : state.InterpolationPoints.Count - 1;

            // Calculate the number of spline points within this segment.
            int segmentPointCount = endPointIndex - startPointIndex;
            if (segmentPointCount <= 0) // Handle cases where there are no points between waypoints.
            {
                state.CurrentPosition = state.InterpolationPoints[startPointIndex];
            }
            else
            {
                // Calculate the current position within the segment's spline points.
                float segmentT = t * segmentPointCount;
                int lerpStartIndex = startPointIndex + Mathf.FloorToInt(segmentT);
                int lerpNextIndex = startPointIndex + Mathf.CeilToInt(segmentT);

                // Clamp the indices to prevent errors at the end of the path.
                lerpStartIndex = Mathf.Clamp(lerpStartIndex, startPointIndex, endPointIndex);
                lerpNextIndex = Mathf.Clamp(lerpNextIndex, startPointIndex, endPointIndex);

                // Interpolate between the two closest spline points in the current segment.
                state.CurrentPosition = Vector3.Lerp(state.InterpolationPoints[lerpStartIndex], state.InterpolationPoints[lerpNextIndex], segmentT - Mathf.FloorToInt(segmentT));
            }

            // Update agent position.
            if (agentObjects.ContainsKey(agentId))
            {
                agentObjects[agentId].transform.position = state.CurrentPosition;
            }

            // Update movement direction for camera.
            Vector3 movementDirection = Vector3.zero;
            if (state.CurrentWaypointIndex + 1 < state.WaypointIndices.Count)
            {
                movementDirection = (state.InterpolationPoints[state.WaypointIndices[state.CurrentWaypointIndex + 1]] - state.CurrentPosition).normalized;
            }

            // Update path visualization for the selected agent.
            if (agentId == selectedAgentId && agentPaths.ContainsKey(agentId))
            {
                LineRenderer lineRenderer = agentPaths[agentId];
                int startIndex = startPointIndex + Mathf.FloorToInt(t * segmentPointCount);
                int remainingPoints = state.InterpolationPoints.Count - startIndex;
                lineRenderer.positionCount = remainingPoints;
                lineRenderer.SetPositions(state.InterpolationPoints.GetRange(startIndex, remainingPoints).ToArray());
            }

            // Advance to next waypoint if interpolation for the current segment is complete.
            if (t >= 1f)
            {
                if (currentWaypointIndex + 1 < state.WaypointIndices.Count - 1)
                {
                    state.CurrentWaypointIndex++;
                    state.InterpolationTime = 0f;
                }
                else
                {
                    // We've reached the final waypoint. Stop interpolating.
                    state.IsInterpolating = false;
                }
            }

            // Notify camera if this is the selected agent.
            if (agentId == selectedAgentId && agentObjects.ContainsKey(agentId))
            {
                OnAgentSelected?.Invoke(agentId, agentObjects[agentId], movementDirection);
            }
        }
    }

    public void SetInitialEnvironmentData(InitialEnvironmentData data)
    {
        initialData = data;
    }

    public void RenderAgents(EpisodeStepData episodeData)
    {
        if (agentPrefab == null || lineRendererMaterial == null || initialData == null || initialData.grid_metadata == null || episodeData == null || episodeData.agents == null)
        {
            Debug.LogError("Required data or prefabs are missing.");
            return;
        }

        int nx = initialData.grid_metadata.nx;
        int ny = initialData.grid_metadata.ny;
        float dx = initialData.grid_metadata.dx;
        float dy = initialData.grid_metadata.dy;

        // Update agent IDs
        agentIds.Clear();
        foreach (var agent in episodeData.agents)
        {
            agentIds.Add(agent.agent_id);
        }

        // Clean up removed agents
        List<int> keysToRemove = agentObjects.Keys.Where(key => !agentIds.Contains(key)).ToList();
        foreach (var key in keysToRemove)
        {
            if (agentObjects.ContainsKey(key)) Destroy(agentObjects[key]);
            if (agentPaths.ContainsKey(key)) Destroy(agentPaths[key].gameObject);
            if (agentTargets.ContainsKey(key)) Destroy(agentTargets[key]);
            agentObjects.Remove(key);
            agentPaths.Remove(key);
            agentTargets.Remove(key);
            agentStates.Remove(key);
            previousPositions.Remove(key);
        }

        // Process each agent
        foreach (var agent in episodeData.agents)
        {
            int agentId = agent.agent_id;
            Vector3 nextPosition = GetAgentPosition(agent, nx, ny, dx, dy);

            // Update or create agent state
            AgentState state = agentStates.ContainsKey(agentId) ? agentStates[agentId] : new AgentState();
            if (!agentStates.ContainsKey(agentId))
            {
                state.CurrentPosition = nextPosition;
                state.NextPosition = nextPosition;
                state.InterpolationTime = stateUpdateInterval;
                state.IsInterpolating = false;
                state.InterpolationPoints = new List<Vector3> { nextPosition };
                state.WaypointIndices = new List<int> { 0 };
                agentStates.Add(agentId, state);
            }
            else
            {
                state.CurrentPosition = agentObjects.ContainsKey(agentId) ? agentObjects[agentId].transform.position : nextPosition;
                state.NextPosition = nextPosition;
                state.InterpolationTime = 0f;
                state.IsInterpolating = true;
                state.CurrentWaypointIndex = 0;
                var (points, indices) = CalculatePathPoints(agent, state.CurrentPosition, nx, ny, dx, dy);
                state.InterpolationPoints = points;
                state.WaypointIndices = indices;
            }
            state.CurrentAgentData = agent;

            // Calculate movement direction
            Vector3 movementDirection = CalculateMovementDirection(agent, nx, ny, dx, dy, nextPosition, agentId);

            // Update or instantiate agent
            GameObject agentObject = agentObjects.ContainsKey(agentId) ? agentObjects[agentId] : Instantiate(agentPrefab, state.CurrentPosition, Quaternion.identity, transform);
            if (!agentObjects.ContainsKey(agentId))
            {
                agentObject.name = $"Agent_{agentId}";
                agentObjects.Add(agentId, agentObject);
            }
            agentObject.SetActive(true);

            // Set up path and target
            RenderAgentPath(agentId, state.InterpolationPoints);
            RenderAgentTarget(agent, nx, ny, dx, dy);

            // Set visibility
            bool isSelected = (agentId == selectedAgentId);
            if (agentPaths.ContainsKey(agentId)) agentPaths[agentId].enabled = isSelected;
            if (agentTargets.ContainsKey(agentId)) agentTargets[agentId].SetActive(isSelected);

            // Notify camera
            if (isSelected) OnAgentSelected?.Invoke(agentId, agentObject, movementDirection);

            previousPositions[agentId] = nextPosition;
        }

        // Select first agent if none selected
        if (selectedAgentId == -1 && agentIds.Count > 0) SelectAgent(agentIds[0]);
    }

    private Vector3 CalculateMovementDirection(AgentData agent, int nx, int ny, float dx, float dy, Vector3 nextPosition, int agentId)
    {
        Vector3 movementDirection = Vector3.zero;
        if (agent._sim.path != null && agent._sim.path.Count > 1 && agent._sim.path[0] != null && agent._sim.path[1] != null)
        {
            float x0 = agent._sim.path[0][0] * (nx * dx) / (nx - 1);
            float z0 = agent._sim.path[0][1] * (ny * dy) / (ny - 1);
            float x1 = agent._sim.path[1][0] * (nx * dx) / (nx - 1);
            float z1 = agent._sim.path[1][1] * (ny * dy) / (ny - 1);
            movementDirection = new Vector3(x1 - x0, 0, z1 - z0).normalized;
        }
        else if (previousPositions.ContainsKey(agentId))
        {
            movementDirection = (nextPosition - previousPositions[agentId]).normalized;
            if (movementDirection.magnitude < 0.01f) movementDirection = Vector3.zero;
        }
        return movementDirection;
    }

    private (List<Vector3>, List<int>) CalculatePathPoints(AgentData agent, Vector3 currentPosition, int nx, int ny, float dx, float dy)
    {
        List<Vector3> controlPoints = new List<Vector3> { currentPosition };
        List<int> waypointIndices = new List<int> { 0 };

        if (agent._sim.path != null)
        {
            foreach (var point in agent._sim.path)
            {
                if (point == null || point.Length < 2) continue;
                float x = point[0] * (nx * dx) / (nx - 1);
                float z = point[1] * (ny * dy) / (ny - 1);
                Vector3 pos = new Vector3(x, GetTerrainHeightAt(new Vector3(x, 0, z)) + pathHeightOffset, z);
                controlPoints.Add(pos);
                waypointIndices.Add(controlPoints.Count - 1);
            }
        }

        if (agent.intent != null && !(agent.intent.sim_x_start == 0 && agent.intent.sim_x_end == 0 && agent.intent.sim_y_start == 0 && agent.intent.sim_y_end == 0))
        {
            float i = (agent.intent.sim_x_start + agent.intent.sim_x_end) / 2f;
            float j = (agent.intent.sim_y_start + agent.intent.sim_y_end) / 2f;
            float x = i * (nx * dx) / (nx - 1);
            float z = j * (ny * dy) / (ny - 1);
            Vector3 targetPos = new Vector3(x, GetTerrainHeightAt(new Vector3(x, 0, z)) + pathHeightOffset, z);
            controlPoints.Add(targetPos);
            waypointIndices.Add(controlPoints.Count - 1);
        }

        List<Vector3> splinePoints = new List<Vector3>();
        List<int> splineWaypointIndices = new List<int> { 0 };

        if (controlPoints.Count == 1)
        {
            splinePoints.Add(controlPoints[0]);
        }
        else if (controlPoints.Count > 1)
        {
            for (int i = 0; i < controlPoints.Count - 1; i++)
            {
                Vector3 p0 = i > 0 ? controlPoints[i - 1] : controlPoints[i];
                Vector3 p1 = controlPoints[i];
                Vector3 p2 = controlPoints[i + 1];
                Vector3 p3 = i < controlPoints.Count - 2 ? controlPoints[i + 2] : p2;

                for (int j = 0; j < segmentsPerPoint; j++)
                {
                    float t = j / (float)segmentsPerPoint;
                    Vector3 point = CatmullRom(p0, p1, p2, p3, t);
                    point.y = GetTerrainHeightAt(point) + pathHeightOffset; // Use pathHeightOffset instead of verticalOffset
                    splinePoints.Add(point);
                }
                if (i < controlPoints.Count - 2)
                {
                    splineWaypointIndices.Add(splinePoints.Count);
                }
            }
            splinePoints.Add(controlPoints[controlPoints.Count - 1]);
            splineWaypointIndices.Add(splinePoints.Count - 1);
        }

        return (splinePoints, splineWaypointIndices);
    }

    private void RenderAgentPath(int agentId, List<Vector3> points)
    {
        if (points == null || points.Count == 0)
        {
            if (agentPaths.ContainsKey(agentId)) Destroy(agentPaths[agentId].gameObject);
            agentPaths.Remove(agentId);
            return;
        }

        LineRenderer lineRenderer = agentPaths.ContainsKey(agentId) ? agentPaths[agentId] : new GameObject($"Path_{agentId}").AddComponent<LineRenderer>();
        if (!agentPaths.ContainsKey(agentId))
        {
            lineRenderer.transform.SetParent(transform);
            lineRenderer.material = lineRendererMaterial;
            lineRenderer.startWidth = 0.1f;
            lineRenderer.endWidth = 0.1f;
            agentPaths.Add(agentId, lineRenderer);
        }

        lineRenderer.positionCount = points.Count;
        lineRenderer.SetPositions(points.ToArray());
        lineRenderer.enabled = (agentId == selectedAgentId);
    }

    private void RenderAgentTarget(AgentData agent, int nx, int ny, float dx, float dy)
    {
        int agentId = agent.agent_id;
        if (agent.intent == null || (agent.intent.sim_x_start == 0 && agent.intent.sim_x_end == 0 &&
                                    agent.intent.sim_y_start == 0 && agent.intent.sim_y_end == 0))
        {
            if (agentTargets.ContainsKey(agentId))
            {
                Destroy(agentTargets[agentId]);
                agentTargets.Remove(agentId);
            }
            return;
        }

        // Calculate the center and size of the target region
        float centerX = (agent.intent.sim_x_start + agent.intent.sim_x_end) / 2f;
        float centerY = (agent.intent.sim_y_start + agent.intent.sim_y_end) / 2f;
        float width = Mathf.Abs(agent.intent.sim_x_end - agent.intent.sim_x_start);
        float height = Mathf.Abs(agent.intent.sim_y_start - agent.intent.sim_y_end);

        // Convert to Unity world coordinates
        float worldX = centerX * (nx * dx) / (nx - 1);
        float worldZ = centerY * (ny * dy) / (ny - 1);
        float worldWidth = width * (nx * dx) / (nx - 1);
        float worldHeight = height * (ny * dy) / (ny - 1);

        // Get terrain height at the center
        Vector3 centerPos = new Vector3(worldX, 0, worldZ);
        float terrainHeight = GetTerrainHeightAt(centerPos);
        Vector3 quadPosition = new Vector3(worldX, terrainHeight + pathHeightOffset, worldZ);

        // --- NEW LOGIC FOR MATERIAL SELECTION AND QUAD MANAGEMENT ---
        GameObject targetObject;
        Material selectedMaterial;

        // Select the material based on the intent type
        Debug.Log("INTENT TYPE: " + agent.intent.type);
        switch (agent.intent.type)
        {
            case 0:
                selectedMaterial = waterMaterial;
                break;
            case 1:
                selectedMaterial = firebreakMaterial;
                break;
            case 2:
                selectedMaterial = retardantMaterial;
                break;
            case 3:
                selectedMaterial = airSupportMaterial;
                break;
            case 4:
                selectedMaterial = idleMaterial;
                break;
            default:
                selectedMaterial = unknownMaterial;
                break;
        }

        if (!agentTargets.ContainsKey(agentId))
        {
            // Validate materials
            if (selectedMaterial == null || outlineMaterial == null)
            {
                Debug.LogError("Missing a required material in the inspector.");
                return;
            }

            // Create a parent object to hold the quads
            targetObject = new GameObject($"Target_Quad_{agentId}");
            targetObject.transform.SetParent(transform);

            // Create outline quad
            GameObject outlineQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            outlineQuad.name = $"Outline_Quad_{agentId}";
            outlineQuad.transform.SetParent(targetObject.transform);
            outlineQuad.GetComponent<Renderer>().material = outlineMaterial;

            // Create the single main quad for visualization
            GameObject mainQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            mainQuad.name = $"Main_Quad_{agentId}";
            mainQuad.transform.SetParent(targetObject.transform);
            mainQuad.GetComponent<Renderer>().material = selectedMaterial; // Assign the material here

            agentTargets.Add(agentId, targetObject);
        }
        else
        {
            targetObject = agentTargets[agentId];

            // Update the main quad's material directly
            Renderer mainQuadRenderer = targetObject.transform.GetChild(1).GetComponent<Renderer>();
            mainQuadRenderer.material = selectedMaterial;
        }

        // Set position, rotation, and scale for both quads
        targetObject.transform.position = quadPosition;
        Transform outlineQuadTransform = targetObject.transform.GetChild(0);
        outlineQuadTransform.localPosition = Vector3.zero;
        outlineQuadTransform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        outlineQuadTransform.localScale = new Vector3(worldWidth * 1.05f, worldHeight * 1.05f, 1f);

        Transform mainQuadTransform = targetObject.transform.GetChild(1);
        mainQuadTransform.localPosition = Vector3.zero;
        mainQuadTransform.localRotation = Quaternion.Euler(90f, 0f, 0f); // Make sure it faces up
        mainQuadTransform.localScale = new Vector3(worldWidth, worldHeight, 1f);

        // Apply a pulse effect to the material instance of the selected agent.
        if (agentId == selectedAgentId)
        {
            Renderer mainQuadRenderer = targetObject.transform.GetChild(1).GetComponent<Renderer>();
            Color baseColor = selectedMaterial.color;
            float pulse = Mathf.Sin(Time.time * 5f) * 0.1f + 0.9f;
            Color pulsedColor = new Color(baseColor.r, baseColor.g, baseColor.b, baseColor.a * pulse);

            // This modifies the color of this specific renderer's material instance,
            // not the shared material asset in the array.
            mainQuadRenderer.material.color = pulsedColor;
        }

        // Set visibility based on selection
        targetObject.SetActive(agentId == selectedAgentId);
    }

    private void SelectAgent(int agentId)
    {
        if (!agentIds.Contains(agentId) || !agentObjects.ContainsKey(agentId)) return;
        selectedAgentId = agentId;
        foreach (var key in agentObjects.Keys)
        {
            bool isSelected = (key == agentId);
            if (agentPaths.ContainsKey(key)) agentPaths[key].enabled = isSelected;
            if (agentTargets.ContainsKey(key)) agentTargets[key].SetActive(isSelected);
        }
        Vector3 movementDirection = previousPositions.ContainsKey(agentId) ? (agentObjects[agentId].transform.position - previousPositions[agentId]).normalized : Vector3.zero;
        OnAgentSelected?.Invoke(agentId, agentObjects[agentId], movementDirection);
    }

    public void SelectNextAgent()
    {
        if (agentIds.Count == 0) return;
        int currentIndex = agentIds.IndexOf(selectedAgentId);
        int nextIndex = (currentIndex + 1) % agentIds.Count;
        SelectAgent(agentIds[nextIndex]);
    }

    public void SelectPreviousAgent()
    {
        if (agentIds.Count == 0) return;
        int currentIndex = agentIds.IndexOf(selectedAgentId);
        int prevIndex = (currentIndex - 1 + agentIds.Count) % agentIds.Count;
        SelectAgent(agentIds[prevIndex]);
    }

    private Vector3 GetAgentPosition(AgentData agent, int nx, int ny, float dx, float dy)
    {
        if (agent._sim.position == null || agent._sim.position.Length < 2) return Vector3.zero;
        float x = agent._sim.position[0] * (nx * dx) / (nx - 1);
        float z = agent._sim.position[1] * (ny * dy) / (ny - 1);
        Vector3 position = new Vector3(x, GetTerrainHeightAt(new Vector3(x, 0, z)) + verticalOffset, z);
        return position;
    }

    private float GetTerrainHeightAt(Vector3 position)
    {
        RaycastHit hit;
        if (Physics.Raycast(new Vector3(position.x, position.y + raycastHeight, position.z), Vector3.down, out hit, raycastHeight * 2f, terrainLayer))
        {
            return hit.point.y;
        }
        return 0f;
    }

    private Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
    }
}