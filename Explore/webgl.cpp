
#define M_PI   3.1415926535897932384626433832795
#define M_2_PI 6.2831853
#define M_PI_2 1.5707963267948966192313216916398
#define EPSILON 1e-15

// Cribbed random number generator.  I do *NOT* use it for important stuff.
float random( vec2 p, out float seed)
{
   // 23.140... e^pi (Gelfond's constant)
   //  2.665... 2^sqrt(2) (Gelfondâ€“Schneider constant)
   vec2 r = vec2( 23.14069263277926, 2.665144142690225 );
   float val = fract( cos( dot( p+seed, r ) ) * 123456. );
   seed = fract(val * 4.8271);
   return val;
}

// Random vec3 using seed.
vec3 v3rand ( vec2 n, out float seed )
{
    return vec3(random(n, seed), random(n, seed), random(n, seed));
}


///////////////////////////////////////////////////////////////////////
// Geometry 

// Construct a matrix which when pre-multiplied computes the cross product with v.
mat3 cross_mat(vec3 v)
{
    // Careful, opengl is column major!  My brain works in row major.
    return transpose(mat3(0, -v.z, v.y,
                          v.z, 0, -v.x,
                          -v.y, v.x, 0));
}

// Construct a matrix which performs the rotation encoded in quaternion
mat3 q_mat(vec4 q)
{
    mat3 cx = cross_mat(q.xyz);
    return mat3(1) + 2.0 * q.w * cx + 2.0 * (cx*cx);
}

// Compose two unit quaternions to compute the product of the rotaion matrices. 
vec4 q_mult(vec4 p, vec4 q) {
    vec3 pv = p.xyz, qv = q.xyz;
    return vec4(p.w * qv + q.w * pv + cross(pv, qv), p.w * q.w - dot(pv, qv));
}

// Convert axis and angle (rotation) into a unit quaternion.
vec4 axis_angle_q(vec3 axis, float phi)
{
    //     "Equation 2.39 in Hartley and Zisserman."
    return vec4(sin(phi*0.5) * normalize(axis), cos(phi*0.5));
}

// Directly apply quaternion rotation to point.
vec3 q_rotate(vec4 qrotor, vec3 point) {
    vec3 rv = qrotor.xyz;
    return q_mult(qrotor, vec4(point * qrotor.w - cross(point, rv), dot(point, rv))).xyz;
}

vec4 slerp(vec3 u0, vec3 u1, float t) {

    return axis_angle_q(cross(u0, u1), t * acos(dot(u0, u1)));
}

///////////////////////////////////////////////////////////////////////////////
// Tests for geometry code.
void testCross( float r, vec3 v1, vec3 v2, out vec3 va, out vec3 vb)
{
    va = cross(v1, v2);
    vb = cross_mat(v1) * v2;
}    
 
// Test that direct quat rotation is equal to matrix rotation computed from quat 
void testRotMat( float r, vec3 v1, vec3 v2, out vec3 va, out vec3 vb)
{    
    vec4 q = axis_angle_q(v1, r);
       
    mat3 rot = q_mat(q);  
    va = rot * v2;
    
    vb = q_rotate(q, v2);
}


// Test that rotating forward and then back works.
void testRot( float r, vec3 v1, vec3 v2, out vec3 va, out vec3 vb)
{    
    vec4 q1 = axis_angle_q(v1, r);
    vec4 q2 = axis_angle_q(v1, -r);
       
    va = v2;
    vb = q_rotate(q1, q_rotate(q2, v2));
}


// Generates random vectors and uses them to test various equalities.
void tester( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalize and center to [-1, 1]
    vec2 nc = (2.0 * fragCoord.xy / iResolution.xy) - vec2(1.0);

    float seed = 0.0;

    float r = random(nc, seed);
    vec3 v1 = v3rand(nc, seed);
    vec3 v2 = v3rand(nc, seed);
    
    vec3 va, vb;

    testRot(r, v1, v2, va, vb);

    vec3 vdiff = va - vb;
    float diff = dot(vdiff, vdiff);
                  
    fragColor = vec4(vdiff * 100000.0, 1.0);
    
    // fragColor = vec4(v2, 1.0);
    // fragColor = vec4(r, r, r, 1.0);
}

float v2max(vec2 v)
{
    return max(v.x, v.y);
}

////////////////////////////////////////////////////////////////////////////
// "Application code"

// Compute a simple camera view from a cube map.
void simpleCamera(out vec4 fragColor, in vec2 fragCoord )
{
    // Create view vectors
    float scale = v2max(iResolution.xy);
    vec2 nc = (2.0 * fragCoord.xy - iResolution.xy) / iResolution.xy;
    
    float fov = radians(120.0);
    
    vec3 ray = vec3(nc * tan(fov/2.0), -1.0);
    
    vec2 mouse = vec2(M_2_PI, M_PI) * (iMouse.xy / iResolution.xy - 0.5);
    
    ray = q_rotate(axis_angle_q(vec3(1., 0., 0.), -mouse.y), ray);
    ray = q_rotate(axis_angle_q(vec3(0., 1., 0.), mouse.x), ray);
    
	fragColor = texture(iChannel0, ray);
    // fragColor = vec4(ray, 1.0);   
}

// Compute a fisheye camera from a cube map.  See 
//  https://www.isprs.org/proceedings/xxxvi/5-W8/Paper/PanoWS_Berlin2005_Schwalbe.pdf
void fisheyeCamera(out vec4 fragColor, in vec2 fragCoord )
{
    // Convert pixels to normalized coordinates of equal size
    float scale = v2max(iResolution.xy);
    vec2 nc = (2.0 * fragCoord.xy - iResolution.xy) / scale;
    
    float fov = radians(180.0);
    
    // from http://paulbourke.net/dome/fisheyecorrect/fisheyerectify.pdf
    // Convert image coordinates to polar coordinates
    float theta = atan(nc.y, nc.x);
    float phi = length(nc) * fov / 2.0;
    
    // Convert to a view ray.  
    vec3 ray = vec3(sin(phi)*cos(theta), sin(phi)*sin(theta), -cos(phi));
    
    // Rotate view space using the mouse
    vec2 mouse = vec2(M_2_PI, M_PI) * (iMouse.xy / iResolution.xy - 0.5);
    
    ray = q_rotate(axis_angle_q(vec3(1., 0., 0.), -mouse.y), ray);
    ray = q_rotate(axis_angle_q(vec3(0., 1., 0.), mouse.x), ray);
    
	fragColor = texture(iChannel0, ray);
}


void cubeMapToEquirectangular(out vec4 fragColor, in vec2 fragCoord )
{
    // Normalize and center to [-1, 1]
    vec2 normCoord = (2.0 * fragCoord.xy / iResolution.xy) - vec2(1.0);

    // Pixels in the resulting image are indexed by longitude (theta) and latitude (phi)
    vec2 thetaphi = normCoord * vec2(M_PI, M_PI_2);
    
    // Phi is elevation. Theta is azimuth.

    // Convert to a ray direction for indexing the cube map.
    vec3 ray = vec3(cos(thetaphi.y) * cos(thetaphi.x),
                             sin(thetaphi.y),
                             cos(thetaphi.y) * sin(thetaphi.x));
    
    vec2 mouse = vec2(M_2_PI, M_PI) * (iMouse.xy / iResolution.xy - 0.5);
    
    ray = q_rotate(axis_angle_q(vec3(0., 0., 1.), mouse.y), ray);
    ray = q_rotate(axis_angle_q(vec3(0., 1., 0.), mouse.x), ray);


	fragColor = texture(iChannel0, ray);
    // fragColor = vec4(vec3(ray.y), 1.0);
}

   
void mainImage(out vec4 fragColor, in vec2 fragCoord )
{
    // fisheyeCamera(fragColor, fragCoord);
    // simpleCamera(fragColor, fragCoord);
    cubeMapToEquirectangular(fragColor, fragCoord);
    // tester(fragColor, fragCoord);
}

