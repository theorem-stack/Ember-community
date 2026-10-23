using UnityEngine;

[System.Serializable]
public class NoiseTextureGenerator : MonoBehaviour
{
    [Header("Noise Settings")]
    [SerializeField] private int textureSize = 256;
    [SerializeField] private float noiseScale = 0.1f;
    [SerializeField] private int octaves = 4;
    [SerializeField] private float persistence = 0.5f;
    [SerializeField] private float lacunarity = 2.0f;
    [SerializeField] private bool seamless = true;

    [Header("Output")]
    [SerializeField] private string fileName = "FireNoiseTexture";

    [ContextMenu("Generate Noise Texture")]
    public Texture2D GenerateNoiseTexture()
    {
        Texture2D noiseTexture = new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, false);

        for (int x = 0; x < textureSize; x++)
        {
            for (int y = 0; y < textureSize; y++)
            {
                float noiseValue = GetNoiseValue(x, y);
                Color pixel = new Color(noiseValue, noiseValue, noiseValue, noiseValue);
                noiseTexture.SetPixel(x, y, pixel);
            }
        }

        noiseTexture.Apply();
        noiseTexture.wrapMode = TextureWrapMode.Repeat;
        noiseTexture.filterMode = FilterMode.Bilinear;

        return noiseTexture;
    }

    private float GetNoiseValue(int x, int y)
    {
        float value = 0f;
        float amplitude = 1f;
        float frequency = noiseScale;

        for (int i = 0; i < octaves; i++)
        {
            float sampleX, sampleY;

            if (seamless)
            {
                // Create seamless noise by sampling on a torus
                float angle1 = (float)x / textureSize * 2f * Mathf.PI;
                float angle2 = (float)y / textureSize * 2f * Mathf.PI;

                sampleX = (Mathf.Sin(angle1) + Mathf.Cos(angle2)) * frequency;
                sampleY = (Mathf.Cos(angle1) + Mathf.Sin(angle2)) * frequency;
            }
            else
            {
                sampleX = x * frequency;
                sampleY = y * frequency;
            }

            float perlinValue = Mathf.PerlinNoise(sampleX, sampleY);
            value += perlinValue * amplitude;

            amplitude *= persistence;
            frequency *= lacunarity;
        }

        // Normalize to 0-1 range
        value = Mathf.Clamp01(value / 2f);

        return value;
    }

    [ContextMenu("Generate and Save Texture")]
    public void GenerateAndSaveNoiseTexture()
    {
        Texture2D texture = GenerateNoiseTexture();


#if UNITY_EDITOR
        // Save as asset in Unity Editor
        string path = $"Assets/{fileName}.asset";
        UnityEditor.AssetDatabase.CreateAsset(texture, path);
        UnityEditor.AssetDatabase.SaveAssets();
        UnityEditor.AssetDatabase.Refresh();

        Debug.Log($"Noise texture saved to {path}");
#else
        Debug.Log("Texture generation complete. Save functionality only available in editor.");
#endif
    }

    // Generate a simple fire-like gradient texture
    [ContextMenu("Generate Fire Gradient Texture")]
    public Texture2D GenerateFireGradientTexture()
    {
        Texture2D fireTexture = new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, false);

        for (int x = 0; x < textureSize; x++)
        {
            for (int y = 0; y < textureSize; y++)
            {
                float normalizedY = (float)y / textureSize;
                float normalizedX = (float)x / textureSize;

                // Create a circular gradient that's stronger at the bottom
                float centerX = 0.5f;
                float centerY = 0.3f; // Offset towards bottom

                float distance = Vector2.Distance(new Vector2(normalizedX, normalizedY), new Vector2(centerX, centerY));
                float radialGradient = 1f - Mathf.Clamp01(distance / 0.6f);

                // Vertical gradient (stronger at bottom)
                float verticalGradient = 1f - normalizedY;

                // Combine gradients
                float alpha = radialGradient * verticalGradient;

                // Add some noise variation
                float noise = Mathf.PerlinNoise(x * 0.05f, y * 0.05f) * 0.3f;
                alpha = Mathf.Clamp01(alpha + noise);

                Color pixel = new Color(1f, 1f, 1f, alpha);
                fireTexture.SetPixel(x, y, pixel);
            }
        }

        fireTexture.Apply();
        fireTexture.wrapMode = TextureWrapMode.Clamp;
        fireTexture.filterMode = FilterMode.Bilinear;

        return fireTexture;
    }

    [ContextMenu("Generate and Save Fire Gradient Texture")]
    public void GenerateAndSaveFireGradientTexture()
    {
        Texture2D texture = GenerateFireGradientTexture();


#if UNITY_EDITOR
        // Save as asset in Unity Editor
        string path = $"Assets/{fileName}.asset";
        UnityEditor.AssetDatabase.CreateAsset(texture, path);
        UnityEditor.AssetDatabase.SaveAssets();
        UnityEditor.AssetDatabase.Refresh();

        Debug.Log($"Noise texture saved to {path}");
#else
        Debug.Log("Texture generation complete. Save functionality only available in editor.");
#endif
    }
}