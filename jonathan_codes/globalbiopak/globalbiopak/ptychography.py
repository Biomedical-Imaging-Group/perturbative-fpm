import torch
import numpy as np
import globalbiopak.linop as gbp
from globalbiopak.polynoms import zernike

class Ptychography:
    def __init__(self, size=200, n_img=16, 
    probe_radius=0.6, probe_shape='gaussian',
    defocus=0,
    device="cpu"
    ):
        self.size = size
        self.probe_radius = probe_radius
        self.probe_shape = probe_shape
        self.defocus = defocus  # probe defocus in the Fourier model
        self.n_img = n_img
        self.device = device

        self.generate_probe()
        self.generate_positions()

    def generate_probe(self):
        x_grid = torch.linspace(-1, 1, self.size)
        y_grid = torch.linspace(-1, 1, self.size)
        x_grid2d, y_grid2d = torch.meshgrid(x_grid, y_grid)

        if self.probe_shape == "gaussian":
            self.probe = 1 / np.sqrt(2*np.pi*self.probe_radius**2) * torch.exp(
                - 1 / 2 / self.probe_radius**2 * 
                (x_grid2d**2 + y_grid2d**2)).to(self.device)
        elif self.probe_shape == "disk":
            self.probe = (x_grid2d**2 + y_grid2d**2 < self.probe_radius**2).double().to(self.device)
        elif self.probe_shape == "random":
            self.probe = torch.randn(self.size, self.size).to(self.device)
            # Blur and downsample
            radius = int(self.size / 8)
            center = int(self.size / 2)
            x = np.linspace(0, self.size, self.size, dtype=np.uint)
            xx,yy = np.meshgrid(x, x)
            mask = ((xx-center)**2 + (yy-center)**2) > radius**2
            fdata = torch.fft.fftshift(torch.fft.fft2(self.probe))
            fdata[mask] = 0
            self.probe = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fdata)))
            self.probe = self.probe / np.sqrt(2*np.pi*self.probe_radius**2) * torch.exp(
                - 1 / 2 / self.probe_radius**2 * 
                (x_grid2d**2 + y_grid2d**2)).to(self.device)
        elif self.probe_shape == "fourier":
            defocus_zernike = zernike(size=self.size, radius=self.probe_radius, index=4, device=self.device)
            mask = (x_grid2d**2 + y_grid2d**2 < self.probe_radius**2).double().to(self.device)
            fourier_probe = mask * torch.exp(1j * defocus_zernike * self.defocus)
            print("test")
            self.probe = torch.fft.ifft2(torch.fft.ifftshift(fourier_probe))
            self.probe = self.probe / torch.norm(self.probe)
        
    def generate_positions(self):
        # probe_xpos are between -1 and 1 (normalized)
        # self.shifts_x are in pixels
        if self.n_img >= 4:
            side = int(np.ceil(np.sqrt(self.n_img)))
            left_origin = -1 + self.probe_radius
            right_origin = 1 - self.probe_radius
            grid_1d = np.linspace(left_origin, right_origin, side)
            probe_xpos, probe_ypos = np.meshgrid(grid_1d, grid_1d)
            probe_xpos = np.ravel(probe_xpos)[:self.n_img]
            probe_ypos = np.ravel(probe_ypos)[:self.n_img]
        elif self.n_img == 3:
            f = self.probe_radius
            probe_xpos = [-0.5*f, 0*f, 0.5*f]
            probe_ypos = [-np.sqrt(3)/6*f, np.sqrt(3)/3*f, -np.sqrt(3)/6*f]
        self.shifts_x = [int(xpos*self.size/2) for xpos in probe_xpos]
        self.shifts_y = [int(ypos*self.size/2) for ypos in probe_ypos]

    def generate_linop(self):
        fft2_op = gbp.LinOpFFT2()
        probe_op = gbp.LinOpMul(self.probe)
        for i_probe in range(self.n_img):
            shift_x = self.shifts_x[i_probe]
            shift_y = self.shifts_y[i_probe]
            partial_linop = gbp.LinOpRoll((-shift_x, -shift_y), (0,1))
            partial_linop = gbp.LinOpComposition(probe_op, partial_linop)
            partial_linop = gbp.LinOpComposition(fft2_op, partial_linop)
            if i_probe == 0:
                self.linop = partial_linop
            else:
                self.linop = gbp.StackLinOp(self.linop, 
                    partial_linop)

    def generate_linop2(self):
        fft2_op = gbp.LinOpFFT2()
        for i_probe in range(self.n_img):
            shift_x = self.shifts_x[i_probe]
            shift_y = self.shifts_y[i_probe]
            shifted_probe = torch.roll(
                self.probe, shifts=(shift_x, shift_y), dims=(0,1))
            probe_mult = gbp.LinOpMul(shifted_probe)
            if i_probe == 0:
                self.linop = gbp.LinOpComposition(fft2_op, probe_mult)
            else:
                self.linop = gbp.StackLinOp(self.linop, 
                    gbp.LinOpComposition(fft2_op, probe_mult))

    def apply(self, x):
        for i_probe in range(self.n_img):
            shift_x = self.shifts_x[i_probe]
            shift_y = self.shifts_y[i_probe]
            shifted_probe = torch.roll(
                self.probe, shifts=(shift_x, shift_y), dims=(0,1))
            partial_intensity = torch.abs(
                torch.fft.fft2(x * shifted_probe)
            )**2
            if i_probe == 0:
                output = partial_intensity
            else:
                output = torch.cat((output, partial_intensity), dim=0)
        return output
    
    def apply_local(self, x, batch_idx):
        for i_iter, i_probe in enumerate(batch_idx):
            shift_x = self.shifts_x[i_probe]
            shift_y = self.shifts_y[i_probe]
            shifted_probe = torch.roll(
                self.probe, shifts=(shift_x, shift_y), dims=(0,1))
            partial_intensity = torch.abs(
                torch.fft.fft2(x * shifted_probe)
            )**2
            if i_iter == 0:
                output = partial_intensity
            else:
                output = torch.cat((output, partial_intensity), dim=0)
        return output

    def gradient_descent(self, y, initial_est=None, n_iter=100,
                        lr=1e-1, method="GD",
                        momentum=0.5, nesterov=True):
        if initial_est is not None:
            x_est = initial_est
        else:
            x_est = torch.ones(self.size, self.size, dtype=torch.complex64).to(self.device)
        x_est.requires_grad = True
        if method == "Adadelta":
            optimizer = torch.optim.Adadelta({x_est}, lr=lr)
        elif method == "Adagrad":
            optimizer = torch.optim.Adagrad({x_est}, lr=lr)
        elif method == "Adam":
            optimizer = torch.optim.Adam({x_est}, lr=lr)
        elif method == "AdamW":
            optimizer = torch.optim.AdamW({x_est}, lr=lr)
        elif method == "ASGD":
            optimizer = torch.optim.ASGD({x_est}, lr=lr)
        elif method == "RMSprop":
            optimizer = torch.optim.RMSprop({x_est}, lr=lr)
        else:
            optimizer = torch.optim.SGD({x_est}, lr=lr, momentum=momentum, nesterov=nesterov)
        loss_history = torch.empty(n_iter)

        for i_iter in range(n_iter):
            optimizer.zero_grad()
            y_est = self.apply(x_est)
            loss = torch.norm(y_est-y)
            loss.backward()
            loss_history[i_iter] = loss.detach().cpu()
            optimizer.step()
        return x_est, loss_history

    def conjugate_gradient(self, y, initial_est=None, n_iter=100, lr=1e-1):
        if initial_est is not None:
            x_est = initial_est
        else:
            x_est = torch.ones(self.size, self.size, dtype=torch.complex64).to(self.device)
        x_est.requires_grad = True
        loss_history = torch.empty(n_iter)

        for i_iter in range(n_iter):
            y_est = self.apply(x_est)
            loss = torch.norm(y_est-y)
            loss.backward()
            loss_history[i_iter] = loss.detach().cpu()
            
            current_grad = x_est.grad
            if i_iter == 0:
                conjugate_direction = - current_grad
                previous_grad = current_grad
            else:
                beta = torch.maximum(torch.Tensor([0]).to(self.device), 
                        torch.real(current_grad.ravel().conj().T @ (current_grad.ravel() - previous_grad.ravel()) / 
                        (previous_grad.ravel().conj().T @ previous_grad.ravel())))
                conjugate_direction = - current_grad + beta * conjugate_direction
                previous_grad = current_grad
            lr = self.line_search(x_est, y, conjugate_direction, 
                                current_grad, initial_loss=loss, initial_lr=lr)
            x_est.grad.zero_()
            with torch.no_grad():
                x_est += lr * conjugate_direction
        return x_est, loss_history
    
    def line_search(self, x_est, y, conjugate_direction, 
                    current_grad, initial_loss, initial_lr=1, c=0.9, tau=0.5):
        with torch.no_grad():
            lr = initial_lr
            m = torch.real(current_grad.ravel().T.conj() @ conjugate_direction.ravel())
            if m >= 0:
                print("There may be a sign error in the conjugate gradient optimization.")
            new_x_est = x_est + lr * conjugate_direction
            new_y_est = self.apply(new_x_est)
            new_loss = torch.norm(new_y_est-y)
            while new_loss > initial_loss + lr * c * m:
                lr = tau * lr
                new_x_est = x_est + lr * conjugate_direction
                new_y_est = self.apply(new_x_est)
                new_loss = torch.norm(new_y_est-y)
        return lr

    def local_gradient_descent(self, y, initial_est=None, n_iter=100,
                            lr=1e-1, method="GD", batch_size=1,
                            momentum=0.5, nesterov=True):
        if initial_est is not None:
            x_est = initial_est
        else:
            x_est = torch.ones(self.size, self.size, dtype=torch.complex64).to(self.device)
        x_est.requires_grad = True
        if method == "Adadelta":
            optimizer = torch.optim.Adadelta({x_est}, lr=lr)
        elif method == "Adagrad":
            optimizer = torch.optim.Adagrad({x_est}, lr=lr)
        elif method == "Adam":
            optimizer = torch.optim.Adam({x_est}, lr=lr)
        elif method == "AdamW":
            optimizer = torch.optim.AdamW({x_est}, lr=lr)
        elif method == "ASGD":
            optimizer = torch.optim.ASGD({x_est}, lr=lr)
        elif method == "RMSprop":
            optimizer = torch.optim.RMSprop({x_est}, lr=lr)
        else:
            optimizer = torch.optim.SGD({x_est}, lr=lr, momentum=momentum, nesterov=nesterov)
        loss_history = torch.empty(n_iter)

        for i_iter in range(n_iter):
            optimizer.zero_grad()
            idx_start = np.mod(batch_size * i_iter, self.n_img)
            idx_end = np.mod(batch_size * (i_iter+1)-1, self.n_img)+1  # the -1+1 is here in case idx_end = self.n_img
            batch_idx = range(idx_start, idx_end)

            y_est = self.apply_local(x_est, batch_idx)
            loss = torch.norm(y_est-y[idx_start*self.size:idx_end*self.size, :])
            loss.backward()
            loss_history[i_iter] = loss.detach().cpu()
            optimizer.step()
        return x_est, loss_history

    def PIE(self, y, initial_est=None, n_iter=100, 
            lr=1e-1, alpha=0.15,
            momentum=0.00, eta=0.2, T=5):
        if initial_est is not None:
            x_est = initial_est
        else:
            x_est = torch.ones(self.size, self.size, dtype=torch.complex64).to(self.device)
        idx_order = np.random.permutation(self.n_img)
        loss_history = torch.empty(n_iter)
        max2 = torch.max(torch.abs(self.probe))**2  # for renormalization
        fft2_op = gbp.LinOpFFT2()
        if momentum > 0:
            past_objects = torch.zeros(self.size, self.size, T).to(self.device)
            velocity = torch.zeros(self.size, self.size).to(self.device)

        for i_iter in range(n_iter):
            for i_img in range(self.n_img):
                idx = idx_order[i_img]
                shift_x = self.shifts_x[idx]
                shift_y = self.shifts_y[idx]
                shifted_probe = torch.roll(
                    self.probe, shifts=(shift_x, shift_y), dims=(0,1))
                probe_mult = gbp.LinOpMul(shifted_probe)
                linop = gbp.LinOpComposition(fft2_op, probe_mult)
                est_fourier_field = linop.apply(x_est)
                corrected_fourier_field = (est_fourier_field / 
                    (torch.abs(est_fourier_field)+1e-5) * 
                    torch.sqrt(y[idx*self.size:(idx+1)*self.size, :]))
                corrected_obj_est = fft2_op.applyAdjoint(corrected_fourier_field)
                x_est = x_est + (lr / (alpha * max2 + 
                    (1-alpha) * torch.abs(shifted_probe)**2) * 
                        shifted_probe.conj() * 
                        (corrected_obj_est - x_est))
            if momentum > 0:
                if i_iter >= T:
                    velocity = eta * velocity + (x_est - past_objects[:,:,0])
                past_objects = torch.roll(past_objects, shifts=-1, dims=2)
                past_objects[:, :, -1] = x_est
                x_est = x_est + momentum * velocity
            # Loss evaluation
            y_est = self.apply(x_est)
            loss_history[i_iter] = torch.norm(y_est - y)
        return x_est, loss_history

    def spectralinit(self, y, n_iter=100, method="Lu"):
        """
        Uses the spectral methods to obtain an initial guess of a random Phase Retrieval problem
        The output is the leading eigenvector of a weighted covariance matrix, computed using power iterations
        For more info, see Montanari & Mondelli, 2017
        
        Parameters:
        - y: intensity measurements
        - n_iter: number of power iterations
        
        Output:
        - x_est: final estimate of the spectral method
        """
        
        self.generate_linop()
        d = self.linop.in_size
        y /= torch.mean(y)  # Normalize intensities
        if method == "Lu":  # if we choose the optimal spectral method by Lu & Li, 2019
            t = torch.maximum(1-1/y, torch.tensor(-1))  # add a threshold at -5 to avoid very large eigenvalues
        else:  # otherwise we use the method from the Wirtinger flow paper, Candès et al, 2015
            t = y

        # Power iterations
        x_est = torch.randn(d, d, dtype=torch.complex64).to(self.device)
        # We first do 10 iterations to detect possible negative eigenvalues
        for i_iter in range(np.minimum(n_iter, 10)):
            # We do not construct the weighted covariance matrix but apply it repeatedly
            x_new = self.linop.apply(x_est)
            x_new = t * x_new
            x_new = self.linop.applyAdjoint(x_new)
            x_est = x_new / torch.norm(x_new)

        # Test if it's a negative eigenvalue
        x_new = self.linop.apply(x_est)
        x_new = t * x_new
        x_new = self.linop.applyAdjoint(x_new)
        corr = torch.real(x_new.ravel().T.conj() @ x_est.ravel())
        if corr < 0:  # if there is a negative eigenvalue, add a regularization
            for i_iter in range(n_iter):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_new = x_new + 1.1*torch.abs(corr)*x_est
                x_est = x_new / torch.norm(x_new)
        else:  # otherwise, finish the power iterations
            for i_iter in range(n_iter - 10):
                x_new = self.linop.apply(x_est)
                x_new = t * x_new
                x_new = self.linop.applyAdjoint(x_new)
                x_est = x_new / torch.norm(x_new)
        return x_est
    
    