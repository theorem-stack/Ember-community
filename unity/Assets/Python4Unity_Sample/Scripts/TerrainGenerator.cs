using UnityEngine;
using System.Collections.Generic;

public class TerrainGenerator : MonoBehaviour
{
    public Material terrainMaterial; // Assign a material for the terrain in the inspector
    public GameObject shrubPrefab; // SpeedTree prefab for SHRUB
    public GameObject deciduousTreePrefab; // SpeedTree prefab for DECIDUOUS_TREE
    public GameObject coniferousTreePrefab; // SpeedTree prefab for CONIFEROUS_TREE
    public GameObject deadVegetationPrefab; // SpeedTree prefab for DEAD_VEGETATION
    public GameObject grassPrefab; // Simple grass model or detail object
    public GameObject waterPrefab; // Water plane or effect
    [SerializeField] private LayerMask terrainLayer; // Layer for the terrain to support raycasting

    [Header("Burn Map System")]
    [SerializeField] private BurnMapManager burnMapManager;
    [SerializeField] private Material vegetationBurnMaterial;
    [SerializeField] public float prefabVerticalOffset = 0.05f;

    public void GenerateTerrain(InitialEnvironmentData initialData)
    {
        // Get the grid dimensions from the initial data
        int nx = initialData.grid_metadata.nx;
        int ny = initialData.grid_metadata.ny;
        float dx = initialData.grid_metadata.dx;
        float dy = initialData.grid_metadata.dy;

        // The height map is now a List of Lists.
        List<List<float>> heightMap = initialData.height_map;

        // Check for valid data
        if (heightMap == null || heightMap.Count == 0 || heightMap[0].Count == 0)
        {
            Debug.LogError("Height map data is empty or malformed.");
            return;
        }

        // --- Mesh Generation ---
        Mesh mesh = new Mesh();

        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uvs = new List<Vector2>();

        // Generate vertices, scaling to match nx * dx and ny * dy
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                float x = i * (nx * dx) / (nx - 1);
                float z = j * (ny * dy) / (ny - 1);
                float y = heightMap[i][j];

                vertices.Add(new Vector3(x, y, z));
                uvs.Add(new Vector2((float)i / (nx - 1), (float)j / (ny - 1)));
            }
        }

        // Generate triangles
        for (int i = 0; i < nx - 1; i++)
        {
            for (int j = 0; j < ny - 1; j++)
            {
                int current = i * ny + j;
                int right = i * ny + j + 1;
                int top = (i + 1) * ny + j;
                int topRight = (i + 1) * ny + j + 1;

                // First triangle
                triangles.Add(current);
                triangles.Add(right);
                triangles.Add(top);

                // Second triangle
                triangles.Add(right);
                triangles.Add(topRight);
                triangles.Add(top);
            }
        }

        // Assign the generated data to the mesh
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uvs.ToArray();

        // Recalculate normals for proper lighting
        mesh.RecalculateNormals();

        // Set up the GameObject in Unity
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            meshFilter = gameObject.AddComponent<MeshFilter>();
        }

        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
        }

        MeshCollider meshCollider = GetComponent<MeshCollider>();
        if (meshCollider == null)
        {
            meshCollider = gameObject.AddComponent<MeshCollider>();
        }

        meshFilter.mesh = mesh;
        meshRenderer.material = terrainMaterial;
        meshCollider.sharedMesh = mesh;

        // Set the terrain layer for raycasting
        if (terrainLayer.value == 0)
        {
            Debug.LogWarning("Terrain layer is not set. Please assign a layer in the Inspector for raycasting compatibility.");
        }
        else
        {
            gameObject.layer = Mathf.RoundToInt(Mathf.Log(terrainLayer.value, 2));
            Debug.Log($"Terrain assigned to layer: {LayerMask.LayerToName(gameObject.layer)}");
        }

        Bounds bounds = mesh.bounds;
        float meshWidth = bounds.size.x;
        float meshHeight = bounds.size.z;

        Debug.Log("Mesh Dimensions: " + meshWidth + " (width) x " + meshHeight + " (height)");
        Debug.Log("Expected Dimensions: " + (nx * dx) + " (width) x " + (ny * dy) + " (height)");

        if (Mathf.Approximately(meshWidth, nx * dx) && Mathf.Approximately(meshHeight, ny * dy))
        {
            Debug.Log("Mesh dimensions match the expected values. ✅");
        }
        else
        {
            Debug.LogError("Mesh dimensions do not match the expected values. ❌");
        }

        Debug.Log("Terrain mesh and collider generated successfully!");
    }

    public void AddVegetation(InitialEnvironmentData initialData)
    {
        // Get the grid dimensions from the initial data
        int nx = initialData.grid_metadata.nx;
        int ny = initialData.grid_metadata.ny;
        float dx = initialData.grid_metadata.dx;
        float dy = initialData.grid_metadata.dy;
        List<List<float>> heightMap = initialData.height_map;
        List<List<int>> vegMap = initialData.veg_map;

        // Check for valid data
        if (vegMap == null || vegMap.Count == 0 || vegMap[0].Count == 0)
        {
            Debug.LogError("Vegetation map data is empty or malformed.");
            return;
        }
        if (heightMap == null || heightMap.Count != vegMap.Count || heightMap[0].Count != vegMap[0].Count)
        {
            Debug.LogError("Height map and vegetation map dimensions do not match.");
            return;
        }

        // Clear existing vegetation (if any) to avoid duplicates
        foreach (Transform child in transform)
        {
            if (child.CompareTag("Vegetation"))
            {
                Destroy(child.gameObject);
            }
        }

        // Ensure the vegetation burn material is set up and has the burn map texture
        if (vegetationBurnMaterial != null && burnMapManager != null)
        {
            // Set terrain dimensions for shader
            vegetationBurnMaterial.SetFloat("_TerrainWidth", nx * dx);
            vegetationBurnMaterial.SetFloat("_TerrainDepth", ny * dy);
            vegetationBurnMaterial.SetTexture("_BurnMap", burnMapManager.burnMap);
        }
        else
        {
            Debug.LogError("vegetationBurnMaterial or burnMapManager is not assigned. Vegetation destruction will not work.");
        }

        // Place vegetation assets based on veg_map
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Get the vegetation type from veg_map
                int vegType = vegMap[i][j];
                float x = i * (nx * dx) / (nx - 1);
                float z = j * (ny * dy) / (ny - 1);
                float y = heightMap[i][j] - prefabVerticalOffset;

                // Select the appropriate prefab based on vegetation type
                GameObject prefab = null;
                switch (vegType)
                {
                    case 3: // SHRUB
                        prefab = shrubPrefab;
                        break;
                    case 4: // DECIDUOUS_TREE
                        prefab = deciduousTreePrefab;
                        break;
                    case 5: // CONIFEROUS_TREE
                        prefab = coniferousTreePrefab;
                        break;
                    case 6: // DEAD_VEGETATION
                        prefab = deadVegetationPrefab;
                        break;
                    case 2: // GRASS
                        prefab = grassPrefab;
                        break;
                    case 0: // WATER
                        prefab = waterPrefab;
                        break;
                    case 1: // BARE_SOIL
                        continue; // No asset needed for bare soil
                    default:
                        Debug.LogWarning($"Unknown vegetation type {vegType} at ({i}, {j})");
                        continue;
                }

                if (prefab != null)
                {
                    // Add small random offset to avoid uniform grid appearance
                    float offsetX = Random.Range(-0.1f * dx, 0.1f * dy);
                    float offsetZ = Random.Range(-0.1f * dy, 0.1f * dy);
                    Vector3 position = new Vector3(x + offsetX, y, z + offsetZ);

                    // Instantiate the prefab
                    GameObject instance = Instantiate(prefab, position, Quaternion.identity, transform);
                    instance.tag = "Vegetation"; // Tag for cleanup

                    // Get the MeshRenderer component
                    MeshRenderer renderer = instance.GetComponent<MeshRenderer>();
                    if (renderer != null && vegetationBurnMaterial != null)
                    {
                        // Get the original material from the prefab to access its textures
                        Material originalMaterial = renderer.sharedMaterial;

                        // Create a new material instance based on our burn material
                        Material newMaterial = new Material(vegetationBurnMaterial);

                        // Set only the _MainTex from the original material
                        if (originalMaterial != null)
                        {
                            newMaterial.SetTexture("_MainTex", originalMaterial.GetTexture("_MainTex"));
                        }

                        // Assign the new material instance to the renderer
                        renderer.material = newMaterial;
                    }

                    // Randomize rotation for visual variety
                    instance.transform.Rotate(0, Random.Range(0, 360), 0);

                    // Adjust scale based on VegetationProperties (optional)
                    if (vegType >= 2 && vegType <= 6) // Apply to GRASS, SHRUB, TREES, DEAD_VEG
                    {
                        float height = GetVegetationHeight(vegType);
                        instance.transform.localScale *= Random.Range(0.8f * height, 1.2f * height);
                    }
                }
            }
        }

        Debug.Log("Vegetation added successfully!");
    }

    private float GetVegetationHeight(int vegType)
    {
        // Map vegetation type to height from VegetationProperties
        switch (vegType)
        {
            case 0: return 0.0f; // WATER
            case 1: return 0.0f; // BARE_SOIL
            case 2: return 0.5f; // GRASS
            case 3: return 0.3f; // SHRUB
            case 4: return 3.0f; // DECIDUOUS_TREE
            case 5: return 6.0f; // CONIFEROUS_TREE
            case 6: return 0.5f; // DEAD_VEGETATION
            default: return 1.0f;
        }
    }
}