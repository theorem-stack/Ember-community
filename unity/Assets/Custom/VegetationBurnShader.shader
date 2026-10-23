Shader "Custom/VegetationBurnShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _DissolveTex ("Dissolve Texture (R)", 2D) = "white" {}
        _BurnMap ("Burn Map", 2D) = "black" {}
        _BurnColor ("Burn Color", Color) = (1, 0, 0, 1)
        _BurnThreshold ("Burn Threshold", Range(0, 1)) = 0.3
        _BurnFade ("Burn Fade", Range(0, 1)) = 0.5
        _DissolveScale ("Dissolve Noise Scale", Float) = 1.0
        // Use an Enum to make the render mode selection user-friendly in the inspector.
        [Enum(Opaque,0,AlphaTest,1,Transparent,2)] _RenderMode ("Render Mode", Float) = 1
        _Cutoff ("Alpha Cutoff", Range(0, 1)) = 0.5
        _TerrainWidth ("Terrain Width", Float) = 256
        _TerrainDepth ("Terrain Depth", Float) = 256
        [Toggle] _DebugMode ("Debug Mode (Show Burn Progress)", Float) = 0
    }

    SubShader
    {
        Tags { "RenderType" = "AlphaTest" "Queue" = "AlphaTest" }
        LOD 200

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha // For Transparent mode
            AlphaToMask On // For AlphaTest mode

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            // ----------------------------------------------------------------------------------
            // STRUCTS
            // ----------------------------------------------------------------------------------
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD1;
            };

            // ----------------------------------------------------------------------------------
            // SAMPLERS & BUFFERS
            // ----------------------------------------------------------------------------------
            sampler2D _MainTex;
            sampler2D _DissolveTex;
            sampler2D _BurnMap;
            float4 _MainTex_ST;
            fixed4 _BurnColor;
            float _BurnThreshold;
            float _BurnFade;
            float _DissolveScale;
            float _RenderMode;
            float _Cutoff;
            float _TerrainWidth;
            float _TerrainDepth;
            float _DebugMode;

            // ----------------------------------------------------------------------------------
            // VERTEX SHADER
            // ----------------------------------------------------------------------------------
            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            // ----------------------------------------------------------------------------------
            // FRAGMENT SHADER
            // ----------------------------------------------------------------------------------
            fixed4 frag (v2f i) : SV_Target
            {
                // Map world position to burn map UVs [0,1]
                float2 burnUV = float2(i.worldPos.x / _TerrainWidth, i.worldPos.z / _TerrainDepth);
                burnUV = saturate(burnUV); // Clamp to prevent out-of-bounds
                
                // R: intensity, G: height, B: burnProgress (formerly fuelVolumeFraction)
                float4 burnData = tex2D(_BurnMap, burnUV); 
                float burnValue = burnData.r;
                float burnProgress = burnData.b; // Use the value directly

                // Debug mode: Show burnProgress
                if (_DebugMode > 0.5)
                {
                    return float4(burnProgress, burnProgress, burnProgress, 1);
                }

                // Base color
                fixed4 col = tex2D(_MainTex, i.uv);

                // Darken the material slightly
                // You can also add a new property for the darkening amount for more control.
                col.rgb *= 0.3; // Adjust this value (e.g., 0.8) to control the darkness

                // Charring
                float charAmount = saturate(burnValue / _BurnThreshold);
                col = lerp(col, _BurnColor, charAmount);

                // --- DISSOLVE LOGIC ---
                // Sample dissolve texture
                float dissolveMask = tex2D(_DissolveTex, i.uv * _DissolveScale).r;

                // Create a smooth dissolve effect based on the burn progress
                float dissolveFade = _BurnFade; 
                float alphaCutoff = burnProgress;

                // This smoothstep function creates a glowing edge for the dissolve effect
                float alphaValue = smoothstep(alphaCutoff - dissolveFade, alphaCutoff, dissolveMask);

                // AlphaTest mode: Discard pixels based on the calculated alpha
                if (_RenderMode == 1.0) // AlphaTest
                {
                    clip(alphaValue - _Cutoff);
                    col.a = 1.0; // Ensure alpha is 1 for non-discarded pixels
                }
                else if (_RenderMode == 2.0) // Transparent
                {
                    col.a *= alphaValue;
                }

                return col;
            }
            ENDCG
        }
    }

    Fallback "Diffuse"
}