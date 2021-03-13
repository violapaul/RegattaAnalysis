#define M_PI   3.1415926535897932384626433832795
#define M_2_PI 6.2831853
#define M_PI_2 1.5707963267948966192313216916398


/* quaternions */

vec4 qmult(vec4 p, vec4 q) {
    vec3 pv = p.xyz, qv = q.xyz;
    return vec4(p.w * qv + q.w * pv + cross(pv, qv), p.w * q.w - dot(pv, qv));
}

vec4 qrotor(vec3 axis, float phi) {
    return vec4(sin(phi*0.5) * axis, cos(phi*0.5));
}

vec4 qmouse(vec4 iMouse, vec3 iResolution, float iTime, float initRotation) {
    vec2 init = vec2(0.5 + 0.25*initRotation * sin(iTime), 0.5 + initRotation * cos(iTime));
    vec2 mouse = mix(init, iMouse.xy / iResolution.xy, step(0.0027, iMouse.y));
    vec4 rotY = qrotor(vec3(0., 1., 0.), M_PI - M_2_PI * mouse.x);
    vec4 rotX = qrotor(vec3(1., 0., 0.), M_PI * mouse.y - M_PI_2);
    return qmult(rotY, rotX);
}

vec3 rotate(vec3 point, vec4 qrotor) {
    vec3 rv = qrotor.xyz;
    return qmult(qrotor, vec4(point * qrotor.w - cross(point, rv), dot(point, rv))).xyz;
}

vec4 slerp(vec3 u0, vec3 u1, float t) {
    return qrotor(cross(u0, u1), t * acos(dot(u0, u1)));
}

////////////////////////////////////////////////////////////////

vec4 quat(in vec3 v, in float a)
{
    return vec4(v * sin(a / 2.0), cos(a / 2.0));
}

vec4 quat_inv(in vec4 q)
{
    return vec4(-q.xyz, q.w);
}

vec4 p2q(in vec3 p)
{
    return vec4(p, 0);
}

vec4 q_mul(in vec4 q1, in vec4 q2)
{
    return vec4(q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y, 
                q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x, 
                q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w, 
                q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z);
}

vec3 rotate(in vec3 p, in vec3 v, in float a)
{
    vec4 q = quat(v, a);
    return q_mul(q_mul(q, p2q(p)), quat_inv(q)).xyz;
}

vec3 rotateq(in vec3 p, in vec4 q)
{
    return q_mul(q_mul(q, p2q(p)), quat_inv(q)).xyz;
}
    
////////////////////////////////////////////////////////////////

void cubeMapToEquirectangular( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalize and center
    vec2 normCoord = (fragCoord.xy / iResolution.xy) - vec2(1.0);

    // Pixels in the resulting image are indexed by longitude (theta) and latitude (phi)
    vec2 thetaphi = 2.0 * normCoord * vec2(M_PI, P_PI_2);

    // Convert to a ray direction for indexing the cube map.
    vec3 rayDirection = vec3(cos(thetaphi.y) * sin(thetaphi.x),
                             sin(thetaphi.y),
                             cos(thetaphi.y) * cos(thetaphi.x));

	fragColor = texture(iChannel0, rayDirection);
}


////////////////////////////////////////////////////////////////




#define M_PI   3.1415926535897932384626433832795
#define M_2_PI 6.2831853
#define M_PI_2 1.5707963267948966192313216916398


mat3 cross_mat(vec3 v)
{
    return mat3(0, -v.z, v.y,
                v.z, 0, -v.x,
                -v.y, v.x, 0);
}

mat3 q_mat(vec4 q){
def q_mat(q):
    "Equation 2.40."
    v, w = q_vw(q)
    cx = cross_mat(v)
    return np.identity(3) + 2 * w * cx + 2 * np.dot(cx, cx)

    mat3 cx = cross_mat(v.xyz);
    return mat3(1)

}

vec4 q_mult(vec4 p, vec4 q) {
    vec3 pv = p.xyz, qv = q.xyz;
    return vec4(p.w * qv + q.w * pv + cross(pv, qv), p.w * q.w - dot(pv, qv));
}

vec4 q_rotor(vec3 axis, float phi) {
    return vec4(sin(phi*0.5) * axis, cos(phi*0.5));
}


vec3 rotate(vec3 point, vec4 qrotor) {
    vec3 rv = qrotor.xyz;
    return qmult(qrotor, vec4(point * qrotor.w - cross(point, rv), dot(point, rv))).xyz;
}

vec4 slerp(vec3 u0, vec3 u1, float t) {
    return qrotor(cross(u0, u1), t * acos(dot(u0, u1)));
}



void cubeMapToEquirectangular( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalize and center to [-1, 1]
    vec2 normCoord = (2.0 * fragCoord.xy / iResolution.xy) - vec2(1.0);

    // Pixels in the resulting image are indexed by longitude (theta) and latitude (phi)
    vec2 thetaphi = normCoord * vec2(M_PI, M_PI_2);
    
    // Phi is elevation. Theta is azimuth.

    // Convert to a ray direction for indexing the cube map.
    vec3 rayDirection = vec3(cos(thetaphi.y) * cos(thetaphi.x),
                             sin(thetaphi.y),
                             cos(thetaphi.y) * sin(thetaphi.x));
                             
    vec4                      
                             

	fragColor = texture(iChannel0, rayDirection);
}




void mainImage(out vec4 fragColor, in vec2 fragCoord )
{
    cubeMapToEquirectangular(fragColor, fragCoord);
}
