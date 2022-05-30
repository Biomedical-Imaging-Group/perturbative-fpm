import torch
import numpy as np
import tomosipo as ts

def generate_random_proj(input_data, num_proj=100, angle_scale=1, SNR=50):
    A, anglesrot, anglesazi, anglessph = generate_random_proj_op(input_data.shape, num_proj=num_proj, angle_scale=angle_scale)
    proj = A(input_data)

    signal_lvl = torch.std(proj)
    noise_lvl = signal_lvl * 10**(-SNR/10)
    noise = torch.randn(proj.shape, device=input_data.device) * noise_lvl
    meas = proj + noise

    return meas, anglesrot, anglesazi, anglessph

def generate_random_proj_fourier(input_data, num_proj=100, angle_scale=1, SNR=50):
    A, anglesrot, anglesazi, anglessph = generate_random_proj_op(input_data.shape, num_proj=num_proj, angle_scale=angle_scale)
    proj = A(input_data)
    fourier_meas0 = torch.abs(torch.fft.fft2(proj, dim=(0, 2)))**2

    signal_lvl = torch.std(fourier_meas0)
    noise_lvl = signal_lvl * 10**(-SNR/10)
    noise = torch.randn(proj.shape, device=input_data.device) * noise_lvl
    meas = fourier_meas0 + noise

    return meas, anglesrot, anglesazi, anglessph

def generate_template_proj(input_data, Nsampl=100, angle_scale=1):
    A, anglesrot, anglesazi, anglessph = generate_template_proj_op(input_data.shape, Nsampl=Nsampl, angle_scale=angle_scale)
    proj = A(input_data)
    return proj, anglesrot, anglesazi, anglessph

def generate_template_proj_fourier(input_data, Nsampl=100, angle_scale=1):
    A, anglesrot, anglesazi, anglessph = generate_template_proj_op(input_data.shape, Nsampl=Nsampl, angle_scale=angle_scale)
    proj = A(input_data)
    template_fourier = torch.abs(torch.fft.fft2(proj, dim=(0, 2)))**2
    return template_fourier, anglesrot, anglesazi, anglessph


def generate_random_proj_op(input_shape, num_proj=100, angle_scale=1):
    # First define projection geometry
    detector_shape = 200
    pixel_size = np.sqrt(3) / detector_shape  # sqrt(3) is the diagonal of the unit cube
    detector_position = (0, -2, 0)

    pg = ts.parallel_vec(
        shape=detector_shape,
        ray_dir=(0, 1, 0),
        det_pos=detector_position,
        det_v=(pixel_size, 0, 0),
        det_u=(0, 0, pixel_size),
    )

    # Then define volume geometry
    volume_size = np.array([1, 1, 1])
    anglesrot = np.random.uniform(0, 2*angle_scale*np.pi, (num_proj, 1))
    anglesazi = np.random.uniform(0, 2*angle_scale*np.pi, (num_proj, 1))
    anglessph = np.random.uniform(0, angle_scale*np.pi, (num_proj, 1))
    axis = np.concatenate(
        (np.cos(anglessph), 
        np.sin(anglessph) * np.sin(anglesazi), 
        np.sin(anglessph) * np.cos(anglesazi)), 
        axis=1
    )
    rot_axis_pos = (0, 0, 0)

    vg0 = ts.volume(
        shape=input_shape,
        pos=(0, 0, 0),
        size=volume_size,
    )
    R = ts.rotate(pos=rot_axis_pos, axis=axis, angles=anglesrot)
    vg = R * vg0.to_vec()

    A = ts.operator(vg, pg)
    return A, anglesrot, anglesazi, anglessph

def generate_template_proj_op(input_shape, Nsampl=20, angle_scale=1):
    detector_shape = 200
    pixel_size = np.sqrt(3) / detector_shape  # sqrt(3) is the diagonal of the unit cube
    detector_position = (0, -2, 0)

    pg = ts.parallel_vec(
        shape=detector_shape,
        ray_dir=(0, 1, 0),
        det_pos=detector_position,
        det_v=(pixel_size, 0, 0),
        det_u=(0, 0, pixel_size),
    )

    volume_size = np.array([1, 1, 1])
    anglesrotlist = np.linspace(0, 2*angle_scale*np.pi, Nsampl)
    anglesazilist = np.linspace(0, 2*angle_scale*np.pi, Nsampl)
    anglessphlist = np.linspace(0, angle_scale*np.pi, Nsampl)
    templanglesrot, templanglesazi, templanglessph = np.meshgrid(anglesrotlist, anglesazilist, anglessphlist)
    templanglesrot = np.reshape(templanglesrot, (Nsampl**3, 1))
    templanglesazi = np.reshape(templanglesazi, (Nsampl**3, 1))
    templanglessph = np.reshape(templanglessph, (Nsampl**3, 1))
    templaxis = np.concatenate(
        (np.cos(templanglessph), 
        np.sin(templanglessph) * np.sin(templanglesazi), 
        np.sin(templanglessph) * np.cos(templanglesazi)), 
        axis=1
    )
    rot_axis_pos = (0, 0, 0)

    vg0 = ts.volume(
        shape=input_shape,
        pos=(0, 0, 0),
        size=volume_size,
    )
    R = ts.rotate(pos=rot_axis_pos, axis=templaxis, angles=templanglesrot)
    vg = R * vg0.to_vec()

    A = ts.operator(vg, pg)
    return A, templanglesrot, templanglesazi, templanglessph

def template_matching(proj_meas, template_proj, preproc=None):
    num_proj = proj_meas.shape[1]
    max_idx = torch.zeros(num_proj)

    if preproc is not None:
        template_proj = preproc(template_proj)
        proj_meas = preproc(proj_meas)

    # mean_to_normalize = torch.mean(template_proj)
    # template_proj = template_proj - mean_to_normalize
    # proj_meas = proj_meas - mean_to_normalize
    for idx in range(num_proj):
        current_proj = proj_meas[:, idx, :]
        current_proj = torch.reshape(current_proj, (current_proj.shape[0], 1, current_proj.shape[0]))
        correlations = torch.sum(template_proj * current_proj, dim=(0, 2))
        max_idx[idx] = torch.argmax(correlations)
    return max_idx

def compute_matching_error(anglesrot, anglessph, anglesazi, templanglesrot, templanglessph, templanglesazi, max_idx):
    num_proj = max_idx.shape[0]
    err = 0
    for idx in range(num_proj):
        target_rot = anglesrot[idx]
        target_sph = anglessph[idx]
        target_azi = anglesazi[idx]
        templ_idx = int(max_idx[idx].numpy())
        templ_rot = templanglesrot[templ_idx]
        templ_sph = templanglessph[templ_idx]
        templ_azi = templanglesazi[templ_idx]
        err += (target_rot-templ_rot)**2 + (target_sph-templ_sph)**2 + (target_azi-templ_azi)**2
    return err / num_proj

def generate_ctf(n_pix=512, defocus=0.2e-7, C_s=0.2e-3, envelope=2):
    """
    Returns a CTF
    Based on the dimensionless expression from Frank's book
    All quantities are expressed in SI units (forget the comments, they are based on the CryoGAN code)
    """
    # Physical parameters
    # defocus = 1e-7  # defocus constant in um
    lambda_e = 2e-12   # electron wavelength in A
    # source: https://www.jeol.co.jp/en/words/emterms/search_result.html?keyword=wavelength%20of%20electron
    C_s = 0.2e-3  # spherical aberration strength in mm
    phi = -0.5  # optional phase shift (phase plate)

    # Discretization parameters
#     n_pix = 512  # number of pixels
    pix_unit = 0.637e-10  # pixel size in nm
    f_unit = 1 / (n_pix * pix_unit)

    n2 = n_pix // 2
    my, mx = torch.meshgrid(torch.arange(-np.floor(n_pix/2), np.ceil(n_pix/2)), 
                            torch.arange(-np.floor(n_pix/2), np.ceil(n_pix/2)))
    r2 = mx**2 + my**2
    
    # Use dimensionless variables (Frank book)
    r2_new = f_unit**2 * r2 * (C_s * lambda_e**3)**0.5
    defocus_new = defocus / (C_s * lambda_e)**0.5

    defocus_term = - np.pi * defocus_new * r2_new
    spherical_term = np.pi / 2 * r2_new**2

    ctf = np.sin(defocus_term + spherical_term + phi) * np.exp(-(r2_new/envelope)**2)
    return ctf