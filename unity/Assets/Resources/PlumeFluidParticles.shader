Shader "Ember/PlumeFluidParticles"
{
    Properties
    {
        _Tint ("Tint", Color) = (1, 1, 1, 1)
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Blend SrcAlpha One
            ZWrite Off
            Cull Off
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.5
            #pragma multi_compile_instancing
            #pragma multi_compile _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON
            #include "UnityCG.cginc"

            struct appdata
            {
                uint vid : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Texture2D<float4> _ParticlePosAge;
            Texture2D<float4> _ParticleAux;
            float4 _Tint;

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                uint corner = v.vid % 6;
                uint index = v.vid / 6;
                float4 posAge = _ParticlePosAge.Load(int3((int)index, 0, 0));
                float4 aux = _ParticleAux.Load(int3((int)index, 0, 0));

                float2 c = float2(-1.0, -1.0);
                if (corner == 1) c = float2(1.0, -1.0);
                else if (corner == 2 || corner == 4) c = float2(1.0, 1.0);
                else if (corner == 5) c = float2(-1.0, 1.0);

                if (posAge.w < 0.0)
                {
                    o.pos = float4(0.0, 0.0, 0.0, 0.0);
                    o.uv = 0.0;
                    o.color = 0.0;
                    return o;
                }

                float life = max(aux.x, 0.0001);
                float age01 = saturate(posAge.w / life);
                float size = aux.y * lerp(0.65, 1.7, age01);

                float3 right = normalize(mul((float3x3)UNITY_MATRIX_I_V, float3(1.0, 0.0, 0.0)));
                float3 up = normalize(mul((float3x3)UNITY_MATRIX_I_V, float3(0.0, 1.0, 0.0)));
                float3 world = posAge.xyz + (right * c.x + up * c.y) * size;
                o.pos = mul(UNITY_MATRIX_VP, float4(world, 1.0));
                o.uv = c * 0.5 + 0.5;

                float heat = saturate(aux.z);
                float3 smoke = lerp(float3(0.18, 0.16, 0.14), float3(0.55, 0.55, 0.55), age01);
                float3 flame = lerp(float3(1.0, 0.32, 0.05), float3(1.0, 0.86, 0.45), heat);
                float3 rgb = lerp(smoke, flame, heat * (1.0 - age01));
                float alpha = (1.0 - age01) * (0.28 + 0.5 * heat);
                o.color = float4(rgb, alpha) * _Tint;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float2 d = i.uv * 2.0 - 1.0;
                float r2 = dot(d, d);
                if (r2 > 1.0)
                    discard;
                float falloff = saturate(1.0 - r2);
                falloff *= falloff;
                return float4(i.color.rgb, i.color.a * falloff);
            }
            ENDHLSL
        }
    }
}
