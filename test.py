import numpy as np

# Define the given data
A = np.array([1, 1, 1])
B = np.array([0, 1, 1])
C = np.array([0, 0, 0])

K_s = np.array([1, 1, 1])  # Specular reflection coefficient
K_d = np.array([1, 1, 0])  # Diffuse reflection coefficient
shininess = 2

C_s = np.array([1, 1, 1])  # Light color (white light)

eye = np.array([-1, 1, -4])  # Camera position

# Compute the normal vector for the triangle (A, B, C)
AB = B - A
AC = C - A
normal = np.cross(AB, AC)
normal = normal / np.linalg.norm(normal)

# Function to compute the diffuse and specular components
def compute_phong_color(light_type="directional", light_pos=None, light_dir=None, normal=None):
    # Normalize vectors
    normal = normal / np.linalg.norm(normal)

    # Camera position (eye)
    eye_dir = eye - A
    eye_dir = eye_dir / np.linalg.norm(eye_dir)

    # Compute the light direction for the directional light case
    if light_type == "directional":
        light_dir = np.array([1, 0, 0])  # Directional light in the x direction
        light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Compute the vector from A to the light source for spotlight case
    if light_type == "spotlight":
        # Calculate the vector from A to light source
        L = light_pos - A
        L = L / np.linalg.norm(L)

        # Compute attenuation factor based on the spotlight direction
        cos_theta = np.dot(L, light_dir)
        attenuation = max(0, cos_theta)**2  # Attenuation based on the angle between light direction and vector

    # Diffuse reflection: I_d = K_d * C_s * max(0, normal . L)
    diffuse = K_d * C_s * max(0, np.dot(normal, light_dir))

    # Specular reflection: I_s = K_s * C_s * (r . v)^shininess
    # Calculate reflection direction: r = 2(normal . L) * normal - L
    r = 2 * np.dot(normal, light_dir) * normal - light_dir
    specular = K_s * C_s * max(0, np.dot(r, eye_dir))**shininess

    # Attenuate the specular component for spotlight
    if light_type == "spotlight":
        specular = specular * attenuation

    # Add ambient light (I_a)
    ambient = np.array([0.1, 0.1, 0.1])

    # Total color at vertex A
    color = ambient + diffuse + specular
    return np.clip(color, 0, 1)  # Ensure color components are between 0 and 1

# Compute color for directional light
color_directional = compute_phong_color(light_type="directional", normal=normal)
print(f"Color at vertex A for directional light: {color_directional}")

# Compute color for spotlight (light source at [-1, 1, -4], pointing in the direction [1, 0, 0])
light_pos_spotlight = np.array([-1, 1, -4])
light_dir_spotlight = np.array([1, 0, 0])  # Spotlight direction
color_spotlight = compute_phong_color(light_type="spotlight", light_pos=light_pos_spotlight, light_dir=light_dir_spotlight, normal=normal)
print(f"Color at vertex A for spotlight: {color_spotlight}")
