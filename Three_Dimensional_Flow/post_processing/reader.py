import os
import sys
import numpy as np
import h5py
import scipy
import scipy.signal

import TurTLE
print(TurTLE.__version__)

import TurTLE_addons
print(TurTLE_addons.__version__)

from TurTLE_addons import NSReader
from HDF5Cache import HDF5Cache


def np_abs2(a):
    return np.real(a * np.conjugate(a))

def get_correlation_function(
        x, y):
    """
    measures delay between x and y signals.
    if y is delayed with respect to x, results are useless
    """
    xx = x - x.mean()
    yy = y - y.mean()
    cc =  scipy.signal.correlate(
            xx, yy, mode = 'full',
            method = 'fft')
    ccx = scipy.signal.correlate(
            xx, xx,
            mode = 'valid',
            method = 'fft')
    ccy = scipy.signal.correlate(
            yy, yy,
            mode = 'valid',
            method = 'fft')
    return cc[cc.shape[0]//2:] / (ccx*ccy)**0.5

class DB2D_reader(NSReader):
    def __init__(
            self,
            read_only_cache = True,
            **kwargs):
        # initialize NSReader
        NSReader.__init__(self, histograms_on = False, **kwargs)

        # read extra parameters
        for kk in ['DB_snapshots_per_file',
                   'niter_DB',
                   'nx_DB',
                   'ny_DB',
                   'nz_DB',
                   'friction_coefficient']:
            self.parameters[kk] =  self.get_data_file()['parameters/' + kk][...]
        self.ratio = self.parameters['nx'] / self.parameters['ny']

        # initialize 2D stat tools
        self.N = self.parameters['nz']
        self.rx = self.get_coord('x')
        self.ry = self.get_coord('y')
        self.k2_DB2D = self.kx[None, :]**2 + self.ky[:, None]**2
        self.k2_DB2D[0, 0] = 1.
        self.bad_indices_DB2D = np.where(self.k2_DB2D >= (self.parameters['ny']/3)**2)
        self.norm_factor_DB2D = self.parameters['nx']*self.parameters['ny']
        self.dsets = ['TrS2',
                      'velocity',
                      'velocity_velocity',
                      'velocity_vorticity',
                      'vorticity_vorticity']
        self.stat_cache = HDF5Cache(
                os.path.join(
                    self.work_dir, self.simname + '_filtered_DB_stat_cache.h5'),
                read_only = read_only_cache)
        self.statistics['Kflow_force_units/L'] = 2*np.pi / (self.parameters['fmode'])
        self.statistics['Kflow_force_units/T'] = (self.statistics['Kflow_force_units/L'] / self.parameters['famplitude'])**0.5
        self.statistics['Kflow_force_units/U'] = self.statistics['Kflow_force_units/L'] / self.statistics['Kflow_force_units/T']
        self.statistics['Kflow_force_units/Reynolds'] = (self.statistics['Kflow_force_units/U']*self.statistics['Kflow_force_units/L'] / self.parameters['nu'])
        print('force-based Reynolds number is ', self.statistics['Kflow_force_units/Reynolds'])
        return None
    def get_kx_ky(
            self,
            kcut = None):
        if type(kcut) == type(None):
            kkx = self.kx
            kky = self.ky
        else:
            kkx = self.kx[:int(np.round(kcut/self.parameters['dkx']))+1]
            kky = np.concatenate([
                self.ky[:int(np.round(kcut)+1)],
                self.ky[1-int(np.round(kcut)):]])
        return kkx, kky
    def r2c(self,
            fieldr):
        return np.fft.rfft2(fieldr, axes = (0, 1))
    def c2r(self,
            fieldk):
        fieldr = np.fft.irfft2(fieldk, axes = (0, 1))
        return fieldr*fieldr.shape[0]*fieldr.shape[1]
    def vorticity_from_velocity(
            self,
            velk,
            mode = 'vorticity'):
        kcut = int(np.round((velk.shape[-2]) * self.parameters['dkx']))
        kkx, kky = self.get_kx_ky(kcut = kcut)
        data = velk.copy()
        #print(kkx, kky)
        if (len(data.shape) == 4):
            kkx = kkx[None, None, :]
            kky = kky[None, :, None]
        elif (len(data.shape) == 3):
            kkx = kkx[None, :]
            kky = kky[:, None]
        #print(data.shape, velk.shape, kky.shape, velk.shape)
        data[..., 0] =   1j*kky*velk[..., 2]
        data[..., 1] = - 1j*kkx*velk[..., 2]
        data[..., 2] = 1j*kkx*velk[..., 1] - 1j*kky*velk[..., 0]
        if mode == 'stream_function':
            kk2 = kkx**2 + kky**2
            kk2[..., 0, 0] = 1.
            data = data[..., 2] / kk2
        return data
    def read_filtered_DB_iteration(
            self,
            iteration = 0,
            dataset = 'velocity',
            kcut = None):
        if dataset == 'vorticity':
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = kcut)
            data = velk.copy()
            kkx, kky = self.get_kx_ky(kcut = kcut)
            data[..., 0] =   1j*kky[:, None]*velk[..., 2]
            data[..., 1] = - 1j*kkx[None, :]*velk[..., 2]
            data[..., 2] = 1j*kkx[None, :]*velk[..., 1] - 1j*kky[:, None]*velk[..., 0]
        elif dataset == 'stream_function':
            vortk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'vorticity',
                    kcut = kcut)
            data = vortk[..., 2].copy()
            kkx, kky = self.get_kx_ky(kcut = kcut)
            kk2 = kkx[None, :]**2 + kky[:, None]**2
            kk2[0, 0] = 1.
            data /= kk2
        elif dataset == 'Pi':
            # transfer from 2D component to fluctuations
            # read velocity and vel_vel tensor at full resolution
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = None)
            velvelk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity_velocity',
                    kcut = None)
            velr = self.c2r(velk)
            velvelr = self.c2r(velvelk)
            ## first compute stress
            upupr = velvelr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upupr[..., jj, ii] -= velr[..., jj]*velr[..., ii]
            upupk = self.r2c(upupr)
            ## now compute divergence of stress
            divupupk = np.zeros(upupk.shape[:-1], upupk.dtype)
            divupupk[..., 0] = (1j*self.kx[None, :]*upupk[..., 0, 0] +
                                1j*self.ky[:, None]*upupk[..., 0, 1])
            divupupk[..., 1] = (1j*self.kx[None, :]*upupk[..., 1, 0] +
                                1j*self.ky[:, None]*upupk[..., 1, 1])
            divupupr = self.c2r(divupupk) / (self.parameters['nx']*self.parameters['ny'])
            ## now compute transfer term
            e_2D_fluct = velr[..., 0]*divupupr[..., 0] + velr[..., 1]*divupupr[..., 1]
            data = self.apply_kcut(
                    array = self.r2c(e_2D_fluct) / e_2D_fluct.size,
                    kcut = kcut)
        elif dataset == 'Pi_fluctuations':
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = None)
            velvelk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity_velocity',
                    kcut = None)
            velr = self.c2r(velk)
            velvelr = self.c2r(velvelk)
            ## first compute stress
            upupr = velvelr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upupr[..., jj, ii] -= velr[..., jj]*velr[..., ii]
            ## now compute velocity gradient
            gradvelk = np.zeros(velvelk.shape, velvelk.dtype)
            gradvelk[..., 0, 0] = 1j*self.kx[None, :]*velk[..., 0]
            gradvelk[..., 0, 1] = 1j*self.kx[None, :]*velk[..., 1]
            gradvelk[..., 1, 0] = 1j*self.ky[:, None]*velk[..., 0]
            gradvelk[..., 1, 1] = 1j*self.ky[:, None]*velk[..., 1]
            gradvelr = self.c2r(gradvelk)
            ## now compute transfer term
            e_2D_fluct = (upupr[..., 0, 0]*gradvelr[..., 0, 0] +
                          upupr[..., 0, 1]*gradvelr[..., 0, 1] +
                          upupr[..., 1, 0]*gradvelr[..., 1, 0] +
                          upupr[..., 1, 1]*gradvelr[..., 1, 1])
            data = self.r2c(e_2D_fluct) / e_2D_fluct.size
            if type(kcut) != type(None):
                data = self.apply_kcut(
                    array = data,
                    kcut = kcut)
        elif dataset == 'E2D':
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = None)
            velr = self.c2r(velk)
            ## now compute transfer term
            E2D = (velr[..., 0]*velr[..., 0] +
                   velr[..., 1]*velr[..., 1])*0.5
            data = self.apply_kcut(
                    array = self.r2c(E2D) / E2D.size,
                    kcut = kcut)
        elif dataset == 'Efluctuations':
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = None)
            velvelk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity_velocity',
                    kcut = None)
            velr = self.c2r(velk)
            velvelr = self.c2r(velvelk)
            ## first compute stress
            upupr = velvelr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upupr[..., jj, ii] -= velr[..., jj]*velr[..., ii]
            Efluctk = self.r2c(upupr[..., 0, 0] + upupr[..., 1, 1])*0.5 / (self.parameters['nx']*self.parameters['ny'])
            data = self.apply_kcut(
                    array = Efluctk,
                    kcut = kcut)
        elif dataset == 'Pi_vorticity':
            # FIXME
            # this needs some dealiasing or something, currently has a lot of small scale fluctuations
            velk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity',
                    kcut = None)
            vortk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'vorticity',
                    kcut = None)
            velvortk = self.read_filtered_DB_iteration(
                    iteration = iteration,
                    dataset = 'velocity_vorticity',
                    kcut = None)
            velr = self.c2r(velk)
            vortr = self.c2r(vortk)
            velvortr = self.c2r(velvortk)
            ## first compute stress
            upopr = velvortr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upopr[..., jj, ii] -= velr[..., jj]*vortr[..., ii]
            upopk = self.r2c(upopr) / (self.parameters['nx']*self.parameters['ny'])
            ## now compute divergence of stress
            divupopk = np.zeros(upopk.shape[:-1], upopk.dtype)
            # only nondiagonal term is non-zero, hence single derivative is picked out from sum
            divupopk[..., 0] = (1j*self.ky[:, None]*(upopk[..., 0, 1] - upopk[..., 1, 0]))
            divupopk[..., 1] = (1j*self.kx[None, :]*(upopk[..., 1, 0] - upopk[..., 0, 1]))
            divupopr = self.c2r(divupopk) / (self.parameters['nx']*self.parameters['ny'])
            ## now compute transfer term
            vorticity_e_2D_fluct = vortr[..., 0]*divupopr[..., 0] + vortr[..., 1]*divupopr[..., 1]
            data = self.r2c(vorticity_e_2D_fluct)
            if type(kcut) != type(None):
                data = self.apply_kcut(
                    array = data,
                    kcut = kcut)
        else:
            DBfile_id = iteration // (self.parameters['niter_DB']*self.parameters['DB_snapshots_per_file'])
            DBfile = h5py.File(os.path.join(self.work_dir, self.simname + '_filtered_DB_{0}.h5'.format(DBfile_id)), 'r')
            if type(kcut) == type(None):
                data = DBfile[dataset + '/complex/{0}'.format(iteration)][...]
            else:
                data = self.apply_kcut(
                        array = DBfile[dataset + '/complex/{0}'.format(iteration)],
                        kcut = kcut)
            DBfile.close()
        return data
    def apply_kcut(
            self,
            array = None,
            kcut = None):
         assert(kcut > 1)
         assert(kcut <= min(self.parameters['ny']/2, self.parameters['dkx']*self.parameters['nx']/2))
         return np.concatenate([
                    array[:int(np.round(kcut)+1), :int(np.round(kcut/self.parameters['dkx']))+1],
                    array[1-int(np.round(kcut)):, :int(np.round(kcut/self.parameters['dkx']))+1]])
    def reset_iterations(
            self,
            iter0 = 10*1024,
            iter1 = None,
            iter_skip = 1024):
        if type(iter1) == type(None):
            iter1 = self.get_data_file()['iteration'][...]
        iterations = range(iter0, iter1, iter_skip)
        self.stat_cache.reset_key(iterations)
        return iterations
    def check_iteration(
            self,
            iteration):
        DBfile_id = iteration // (self.parameters['niter_DB']*self.parameters['DB_snapshots_per_file'])
        DBfile_name = os.path.join(self.work_dir, self.simname + '_filtered_DB_{0}.h5'.format(DBfile_id))
        if not os.path.exists(DBfile_name):
            return None
        DBfile = h5py.File(os.path.join(self.work_dir, self.simname + '_filtered_DB_{0}.h5'.format(DBfile_id)), 'r')
        if '{0}'.format(iteration) in DBfile['velocity/complex'].keys():
            DBfile.close()
            return iteration
        else:
            return None
    def check_iterations(
            self,
            iterations):
        new_iterations = []
        for iteration in iterations:
            DBfile_id = iteration // (self.parameters['niter_DB']*self.parameters['DB_snapshots_per_file'])
            DBfile_name = os.path.join(self.work_dir, self.simname + '_filtered_DB_{0}.h5'.format(DBfile_id))
            if not os.path.exists(DBfile_name):
                continue
            DBfile = h5py.File(os.path.join(self.work_dir, self.simname + '_filtered_DB_{0}.h5'.format(DBfile_id)), 'r')
            if '{0}'.format(iteration) in DBfile['velocity/complex'].keys():
                new_iterations.append(iteration)
            DBfile.close()
        return new_iterations
    def read_filtered_DB_iterations(
            self,
            field = 'velocity',
            iter0 = 10*1024,
            iter1 = None,
            iter_skip = 1024,
            kcut = 2,
            return_iterations = False):
        iterations = self.reset_iterations(iter0 = iter0, iter1 = iter1, iter_skip = iter_skip)
        cached_field_name = field + '_kcut{0}'.format(kcut)
        if cached_field_name not in self.stat_cache.keys():
            # check for available iterations
            iterations = self.check_iterations(iterations)
            self.stat_cache.reset_key(iterations)
        cached_field_name = field + '_kcut{0}'.format(kcut)
        if cached_field_name not in self.stat_cache.keys():
            # we need to read data, compute average, save in cache
            val = self.read_filtered_DB_iteration(
                    iterations[0],
                    dataset = field,
                    kcut = kcut)
            data = np.zeros((len(iterations),) + val.shape, val.dtype)
            data[0] = val
            for ii in range(1, len(iterations)):
                print('reading iteration ', iterations[ii])
                new_val = self.read_filtered_DB_iteration(
                        iterations[ii],
                        dataset = field,
                        kcut = kcut)
                data[ii] = new_val
            self.stat_cache[cached_field_name + '/complex'] = data
        else:
            # whether or not individual iterations are still available is irrelevant,
            # we just read the cached data
            data = self.stat_cache[cached_field_name + '/complex'][()]
        if return_iterations:
            return data, iterations
        else:
            return data
    def compute_taverage(
            self,
            field = 'velocity',
            iter0 = 10*1024,
            iter1 = None,
            iter_skip = 1024,
            kcut = 64,
            force = False):
        iterations = self.reset_iterations(iter0 = iter0, iter1 = iter1, iter_skip = iter_skip)
        cached_field_name = field + '_kcut{0}_taverage'.format(kcut)
        if cached_field_name not in self.stat_cache.keys() or force:
            val = self.read_filtered_DB_iteration(
                    iterations[0],
                    dataset = field,
                    kcut = kcut)
            for ii in range(1, len(iterations)):
                print('reading iteration ', iterations[ii])
                val += self.read_filtered_DB_iteration(
                        iterations[ii],
                        dataset = field,
                        kcut = kcut)
            val /= len(iterations)
            if cached_field_name in self.stat_cache.keys():
                del self.stat_cache[cached_field_name + '/complex']
            self.stat_cache[cached_field_name + '/complex'] = val
        else:
            val = self.stat_cache[cached_field_name + '/complex'][()]
        return val
    def compute_Kflow_stats(
            self,
            iter0 = 10*1024,
            iter1 = None,
            iter_skip = 1024,
            force = False):
        iterations = self.reset_iterations(iter0 = iter0, iter1 = iter1, iter_skip = iter_skip)
        cached_group_name = 'Kflow_stats'
        if (cached_group_name not in self.stat_cache.keys()) or force:
            pp_file = self.get_data_file()
            good_indices = np.array(iterations) // self.parameters['niter_stat']
            good_indices_DB = np.array(iterations) // self.parameters['niter_DB']
            Etotal = pp_file['statistics/moments/velocity'][good_indices, 2, 3]/2
            Efluctuations = pp_file['statistics/moments/velocity_fluctuations'][good_indices_DB, 2, 3]/2
            Emeanz = (pp_file['statistics/moments/velocity'][good_indices, 2, 2]/2 -
                      pp_file['statistics/moments/velocity_fluctuations'][good_indices_DB, 2, 2]/2)
            E2D = Etotal - Efluctuations - Emeanz
            diss = self.parameters['nu']*pp_file['statistics/moments/vorticity'][good_indices, 2, 3]
            Urms = np.sqrt(2*np.mean(Efluctuations)/3)
            Lint = Urms**3 / np.mean(diss)
            if cached_group_name in self.stat_cache.keys():
                del self.stat_cache[cached_group_name]
            self.stat_cache[cached_group_name + '/Lint'] = Lint
            self.stat_cache[cached_group_name + '/Uint'] = Urms
            self.stat_cache[cached_group_name + '/Tint'] = Lint / Urms
            self.stat_cache[cached_group_name + '/diss(t)'] = diss
            self.stat_cache[cached_group_name + '/Etotal(t)'] = Etotal
            self.stat_cache[cached_group_name + '/E2D(t)'] = E2D
            self.stat_cache[cached_group_name + '/Emeanz(t)'] = Emeanz
            self.stat_cache[cached_group_name + '/Efluctuations(t)'] = Efluctuations
            self.stat_cache[cached_group_name + '/t'] = np.array(iterations)*self.parameters['dt']
            self.stat_cache[cached_group_name + '/vel_fluctuations/m2(t)'] = pp_file['statistics/moments/velocity_fluctuations'][good_indices_DB, 2, :3]
            self.stat_cache[cached_group_name + '/vel_fluctuations/m4(t)'] = pp_file['statistics/moments/velocity_fluctuations'][good_indices_DB, 4, :3]
            self.stat_cache[cached_group_name + '/vel_fluctuations/m6(t)'] = pp_file['statistics/moments/velocity_fluctuations'][good_indices_DB, 6, :3]
            self.stat_cache[cached_group_name + '/vort_fluctuations/m2(t)'] = pp_file['statistics/moments/vorticity_fluctuations'][good_indices_DB, 2, :3]
            self.stat_cache[cached_group_name + '/vort_fluctuations/m4(t)'] = pp_file['statistics/moments/vorticity_fluctuations'][good_indices_DB, 4, :3]
            self.stat_cache[cached_group_name + '/vort_fluctuations/m6(t)'] = pp_file['statistics/moments/vorticity_fluctuations'][good_indices_DB, 6, :3]
            pp_file.close()
        for kk in ['Lint', 'Tint', 'Uint',
                   'diss(t)', 'Etotal(t)', 'E2D(t)', 'Emeanz(t)', 'Efluctuations(t)', 't']:
            self.statistics['Kflow_stats/' + kk] = self.stat_cache[cached_group_name + '/' + kk][()]
        for mm in [2, 4, 6]:
            for field in ['vel', 'vort']:
                self.statistics['Kflow_stats/' + field + '_fluctuations/m{0}(t)'.format(mm)] = self.stat_cache[cached_group_name + '/' + field + '_fluctuations/m{0}(t)'.format(mm)][()]
        self.statistics['Kflow_stats/small_scale_dissipation(t)'] = np.sum(self.statistics['Kflow_stats/vort_fluctuations/m2(t)'], axis = 1)*self.parameters['nu']
        for kk in ['Etotal', 'E2D', 'Emeanz', 'Efluctuations', 'diss', 'vort_fluctuations/m2', 'vort_fluctuations/m4', 'small_scale_dissipation']:
            self.statistics['Kflow_stats/' + kk + '_tmean'] = np.mean(self.statistics['Kflow_stats/' + kk + '(t)'])
            self.statistics['Kflow_stats/' + kk + '_tvariance'] = np.mean(
                    (self.statistics['Kflow_stats/' + kk + '(t)'] -
                     self.statistics['Kflow_stats/' + kk + '_tmean'])**2)
        return None
    def compute_Kflow_energetics_single_iteration(
            self,
            iteration = 0,
            force = False):
        iterations = self.reset_iterations(iter0 = iteration, iter1 = iteration+1, iter_skip = 1)
        cached_group_name = 'Kflow_energetics_from_slice'
        if (cached_group_name not in self.stat_cache.keys()) or force:
            # read z-averaged velocity
            print('computing Kflow energetics single iteration from slice data, iteration', iteration)
            velk = self.read_filtered_DB_iteration(
                iteration = iteration,
                dataset = 'velocity')
            # put z-averaged velocity in real space
            velr = self.c2r(velk)
            E2D = 0.5*np.mean(velr[..., 0]**2 + velr[..., 1]**2)
            Emeanz = 0.5*np.mean(velr[..., 2]**2)
            # injection rate
            e_F = -self.parameters['famplitude']*0.5*(velk[1, 0, 0].imag - velk[-1, 0, 0].imag)
            # losses of 2D field to drag term
            e_mu = 2*self.parameters['friction_coefficient']*(
                    np_abs2(velk[0, 1, 0]) + np_abs2(velk[0, 2, 0]) +
                    np_abs2(velk[0, 1, 1]) + np_abs2(velk[0, 2, 1]))
            # compute gradient
            kgradU = np.zeros(velk.shape[:2] + (2, 2), velk.dtype)
            kx, ky = self.get_kx_ky()
            kgradU[..., 0, 0] = 1j*kx[None, :]*velk[..., 0]
            kgradU[..., 0, 1] = 1j*ky[:, None]*velk[..., 0]
            kgradU[..., 1, 0] = 1j*kx[None, :]*velk[..., 1]
            kgradU[..., 1, 1] = 1j*ky[:, None]*velk[..., 1]
            rgradU2D = self.c2r(kgradU)
            # losses of 2D field to dissipation
            e_nu = self.parameters['nu']*np.mean(
                    np.sum(rgradU2D**2, axis = (2, 3)))
            # losses of mean z component to dissipation
            kgradUz = np.zeros(velk.shape[:2] + (1, 2), velk.dtype)
            kgradUz[..., 0, 0] = 1j*kx[None, :]*velk[..., 2]
            kgradUz[..., 0, 1] = 1j*ky[:, None]*velk[..., 2]
            rgradUz = self.c2r(kgradUz)
            diss_meanz_component = self.parameters['nu']*np.mean(
                    np.sum(rgradUz**2, axis = (2, 3)))
            # read uu
            velvelk = self.read_filtered_DB_iteration(
                iteration = iteration,
                dataset = 'velocity_velocity')
            velvelr = self.c2r(velvelk)
            Etotal = 0.5*np.mean(velvelr[..., 0, 0] + velvelr[..., 1, 1] + velvelr[..., 2, 2])
            Efluctuations = Etotal - Emeanz - E2D
            # transfer from 2D component to fluctuations
            ## first compute stress
            upupr = velvelr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upupr[..., jj, ii] -= velr[..., jj]*velr[..., ii]
            upupk = self.r2c(upupr)
            ## now compute divergence of stress
            divupupk = np.zeros(upupk.shape[:-1], upupk.dtype)
            divupupk[..., 0] = (1j*self.kx[None, :]*upupk[..., 0, 0] +
                                1j*self.ky[:, None]*upupk[..., 0, 1])
            divupupk[..., 1] = (1j*self.kx[None, :]*upupk[..., 1, 0] +
                                1j*self.ky[:, None]*upupk[..., 1, 1])
            divupupr = self.c2r(divupupk)
            TrS2k = self.read_filtered_DB_iteration(
                iteration = iteration,
                dataset = 'TrS2',
                kcut = 2)
            vortvortk = self.read_filtered_DB_iteration(
                iteration = iteration,
                dataset = 'vorticity_vorticity',
                kcut = 2)
            ## now compute transfer term
            e_2D_fluct = np.mean(velr[..., 0]*divupupr[..., 0] + velr[..., 1]*divupupr[..., 1])
            # transfer from fluctuations to mean z component $\\langle \langle u_z' \bs{u}'\rangle_z \cdot \nabla w \rangle_{xy}$
            ## gradient of mean z component is present in rgradUz
            ## uzprime uprime is contained in velvelr
            upzupx = velvelr[..., 2, 0] - velr[..., 2]*velr[..., 0]
            upzupy = velvelr[..., 2, 1] - velr[..., 2]*velr[..., 1]
            xi_w = np.mean(upzupx*rgradUz[..., 0, 0] + upzupy*rgradUz[..., 0, 1])
            if cached_group_name in self.stat_cache.keys():
                del self.stat_cache[cached_group_name]
            self.stat_cache[cached_group_name + '/Etotal'] = Etotal
            self.stat_cache[cached_group_name + '/E2D'] = E2D
            self.stat_cache[cached_group_name + '/Efluctuations'] = Efluctuations
            self.stat_cache[cached_group_name + '/Emeanz'] = Emeanz
            self.stat_cache[cached_group_name + '/e_F'] = e_F
            self.stat_cache[cached_group_name + '/e_mu'] = e_mu
            self.stat_cache[cached_group_name + '/e_nu'] = e_nu
            self.stat_cache[cached_group_name + '/pi'] = e_2D_fluct / (self.parameters['nx']*self.parameters['ny'])
            self.stat_cache[cached_group_name + '/xi_w'] = xi_w
            self.stat_cache[cached_group_name + '/diss_meanz_component'] = diss_meanz_component
            self.stat_cache[cached_group_name + '/total_dissipation'] = 2*self.parameters['nu']*np.real(TrS2k[0, 0])
            self.stat_cache[cached_group_name + '/total_dissipation2'] = self.parameters['nu']*np.real(vortvortk[0, 0, 0, 0] +
                                                                                                       vortvortk[0, 0, 1, 1] +
                                                                                                       vortvortk[0, 0, 2, 2])
        data = {}
        for kk in ['Etotal', 'E2D', 'Emeanz', 'Efluctuations',
                   'e_F', 'e_mu', 'e_nu', 'pi', 'xi_w',
                   'total_dissipation', 'total_dissipation2', 'diss_meanz_component']:
            data[kk] = self.stat_cache[cached_group_name + '/' + kk][()]
        return data
    def compute_Kflow_energetics(
            self,
            iteration_list = None,
            force = False):
        all_data = [self.compute_Kflow_energetics_single_iteration(iteration = ii, force = force)
                    for ii in iteration_list]
        data = {}
        for kk in all_data[0].keys():
            data[kk] = np.array([all_data[ii][kk] for ii in range(len(all_data))])
        data['t'] = np.array(iteration_list) * self.parameters['dt']
        data['I1'], data['I2'], data['I3'] = self.compute_I123(
                iter0 = iteration_list[0],
                iter1 = iteration_list[-1]+1024,
                iter_skip = 1024)
        data['ks'] = data['I1']/3 + 2*data['I2']/3 + data['I3']
        return data
    def compute_Kflow_energetics_single_iteration_from_checkpoint(
            self,
            iteration = 0,
            force = False):
        iterations = self.reset_iterations(iter0 = iteration, iter1 = iteration+1, iter_skip = 1)
        cached_group_name = 'Kflow_energetics_from_checkpoint'
        if (cached_group_name not in self.stat_cache.keys()) or force:
            # read z-averaged velocity
            cp_file = h5py.File(self.get_checkpoint_fname(iteration = iteration), 'r')
            full_velr = cp_file['velocity/real/{0}'.format(iteration)][()]
            velr = np.mean(full_velr, axis = 0)
            velk = self.r2c(velr)
            # energy of 2D component
            E2D = 0.5*np.mean(velr[..., 0]**2 + velr[..., 1]**2)
            # energy of z-averaged z-velocity
            Emeanz = 0.5*np.mean(velr[..., 2]**2)
            # injection rate
            e_F = -self.parameters['famplitude']*0.5*(velk[1, 0, 0].imag - velk[-1, 0, 0].imag)
            # losses of 2D field to drag term
            e_mu = 2*self.parameters['friction_coefficient']*(
                    np_abs2(velk[0, 1, 0]) + np_abs2(velk[0, 2, 0]) +
                    np_abs2(velk[0, 1, 1]) + np_abs2(velk[0, 2, 1]))
            # compute gradient
            kgradU = np.zeros(velk.shape[:2] + (2, 2), velk.dtype)
            kx, ky = self.get_kx_ky()
            kgradU[..., 0, 0] = 1j*kx[None, :]*velk[..., 0]
            kgradU[..., 0, 1] = 1j*ky[:, None]*velk[..., 0]
            kgradU[..., 1, 0] = 1j*kx[None, :]*velk[..., 1]
            kgradU[..., 1, 1] = 1j*ky[:, None]*velk[..., 1]
            rgradU2D = self.c2r(kgradU)
            # losses of 2D field to dissipation
            e_nu = self.parameters['nu']*np.mean(
                    np.sum(rgradU2D**2, axis = (2, 3)))
            # read uu
            full_velvelr = full_velr[..., :, None]*full_velr[..., None, :]
            velvelr = np.mean(full_velvelr, axis = 0)
            # total energy
            Etotal = 0.5*np.mean(velvelr[..., 0, 0] + velvelr[..., 1, 1] + velvelr[..., 2, 2])
            # energy of fluctuations
            Efluctuations = Etotal - Emeanz - E2D
            # transfer from 2D component to fluctuations
            ## first compute stress
            upupr = velvelr[..., :2, :2].copy()
            for jj in range(2):
                for ii in range(2):
                    upupr[..., jj, ii] -= velr[..., jj]*velr[..., ii]
            upupk = self.r2c(upupr)
            ## now compute divergence of stress
            divupupk = np.zeros(upupk.shape[:-1], upupk.dtype)
            divupupk[..., 0] = (1j*self.kx[None, :]*upupk[..., 0, 0] +
                                1j*self.ky[:, None]*upupk[..., 0, 1])
            divupupk[..., 1] = (1j*self.kx[None, :]*upupk[..., 1, 0] +
                                1j*self.ky[:, None]*upupk[..., 1, 1])
            divupupr = self.c2r(divupupk)
            ## now compute transfer term
            e_2D_fluct = np.mean(velr[..., 0]*divupupr[..., 0] + velr[..., 1]*divupupr[..., 1])
            # store results
            if cached_group_name in self.stat_cache.keys():
                del self.stat_cache[cached_group_name]
            self.stat_cache[cached_group_name + '/Etotal'] = Etotal
            self.stat_cache[cached_group_name + '/E2D'] = E2D
            self.stat_cache[cached_group_name + '/Efluctuations'] = Efluctuations
            self.stat_cache[cached_group_name + '/Emeanz'] = Emeanz
            self.stat_cache[cached_group_name + '/e_F'] = e_F
            self.stat_cache[cached_group_name + '/e_mu'] = e_mu
            self.stat_cache[cached_group_name + '/e_nu'] = e_nu
            self.stat_cache[cached_group_name + '/pi'] = e_2D_fluct / (self.parameters['nx']*self.parameters['ny'])
        data = {}
        for kk in ['Etotal', 'E2D', 'Emeanz', 'Efluctuations',
                   'e_F', 'e_mu', 'e_nu', 'pi']:
            data[kk] = self.stat_cache[cached_group_name + '/' + kk][()]
        return data
    def compute_ks(
            self,
            **kwargs):
        I1, I2, I3 = self.compute_I123(**kwargs)
        return I1/3 + 2*I2/3 + I3
    def compute_I123(
            self,
            **kwargs):
        velk = self.read_filtered_DB_iterations(
                field = 'velocity',
                kcut = 2,
                **kwargs)
        streamk = self.vorticity_from_velocity(velk, mode = 'stream_function')
        assert((len(streamk.shape) == 3))
        e1 = np_abs2(streamk[:, 0, 1])
        e2 = np_abs2(streamk[:, 0, 2])
        e3 = np_abs2(streamk[:, 0, 3])
        return e1 / (e1 + e2 + e3), e2 / (e1 + e2 + e3),  e3 / (e1 + e2 + e3)
    def compute_transfer_rates_correlations(
            self,
            iter0 = 0,
            iter1 = None,
            iter_skip = 1024,
            force = False):
        iterations = self.reset_iterations(iter0 = iter0, iter1 = iter1, iter_skip = iter_skip)
        cached_group_name = 'transfer_rates_cross_correlation_times'
        if (cached_group_name not in self.stat_cache.keys()) or force:
            dd = self.compute_Kflow_energetics(iteration_list = iterations)
            self.compute_Kflow_stats(iter0 = iterations[0], iter1 = iterations[-1]+iter_skip, iter_skip = iter_skip)
            # time reset to 0, to get correlation time
            tt = dd['t'] - dd['t'][0]
            # transfer from 2D to small scales dependency on injection rate
            cf_pi_vs_e_F = get_correlation_function(dd['pi'], dd['e_F'])
            # small scale dissipation dependency on injection rate
            cf_ssd_vs_e_F = get_correlation_function(self.statistics['Kflow_stats/small_scale_dissipation(t)'], dd['e_F'])
            # small scale dissipaiton dependency on transfer from 2D to small scales
            cf_ssd_vs_pi = get_correlation_function(self.statistics['Kflow_stats/small_scale_dissipation(t)'], dd['pi'])
            bla = np.argmax(cf_pi_vs_e_F)
            tmax_pi_vs_e_F = tt[bla]
            vmax_pi_vs_e_F = cf_pi_vs_e_F[bla]
            bla = np.argmax(cf_ssd_vs_e_F)
            tmax_ssd_vs_e_F = tt[bla]
            vmax_ssd_vs_e_F = cf_ssd_vs_e_F[bla]
            bla = np.argmax(cf_ssd_vs_pi)
            tmax_ssd_vs_pi = tt[bla]
            vmax_ssd_vs_pi = cf_ssd_vs_pi[bla]
            # small scale dissipation dependency on 2D energy
            cf_ssd_vs_E2D = get_correlation_function(self.statistics['Kflow_stats/small_scale_dissipation(t)'], dd['E2D'])
            bla = np.argmax(cf_ssd_vs_E2D)
            tmax_ssd_vs_E2D = tt[bla]
            vmax_ssd_vs_E2D = cf_ssd_vs_E2D[bla]
            # small scale dissipation dependency on fluctuation energy
            cf_ssd_vs_Efluctuations = get_correlation_function(self.statistics['Kflow_stats/small_scale_dissipation(t)'], dd['Efluctuations'])
            bla = np.argmax(cf_ssd_vs_Efluctuations)
            tmax_ssd_vs_Efluctuations = tt[bla]
            vmax_ssd_vs_Efluctuations = cf_ssd_vs_Efluctuations[bla]
            # Efluctuations vs energy injection
            cf_Efluctuations_vs_e_F = get_correlation_function(dd['Efluctuations'], dd['e_F'])
            bla = np.argmax(cf_Efluctuations_vs_e_F)
            tmax_Efluctuations_vs_e_F = tt[bla]
            vmax_Efluctuations_vs_e_F = cf_Efluctuations_vs_e_F[bla]
            # E2D vs energy injection
            cf_E2D_vs_e_F = get_correlation_function(dd['E2D'], dd['e_F'])
            bla = np.argmax(cf_E2D_vs_e_F)
            tmax_E2D_vs_e_F = tt[bla]
            vmax_E2D_vs_e_F = cf_E2D_vs_e_F[bla]
            # Efluctuations vs E2D
            cf_Efluctuations_vs_E2D = get_correlation_function(dd['Efluctuations'], dd['E2D'])
            bla = np.argmax(cf_Efluctuations_vs_E2D)
            tmax_Efluctuations_vs_E2D = tt[bla]
            vmax_Efluctuations_vs_E2D = cf_Efluctuations_vs_E2D[bla]
            if cached_group_name in self.stat_cache.keys():
                del self.stat_cache[cached_group_name]
            self.stat_cache[cached_group_name + '/t'] = tt
            self.stat_cache[cached_group_name + '/transfer_2D_small_scales'] = dd['pi']
            self.stat_cache[cached_group_name + '/injection_rate'] = dd['e_F']
            self.stat_cache[cached_group_name + '/E2D'] = dd['E2D']
            self.stat_cache[cached_group_name + '/Efluctuations'] = dd['Efluctuations']
            self.stat_cache[cached_group_name + '/small_scale_dissipation'] = self.statistics['Kflow_stats/small_scale_dissipation(t)']
            self.stat_cache[cached_group_name + '/correlation_transfer_injection'] = cf_pi_vs_e_F
            self.stat_cache[cached_group_name + '/tmax_correlation_transfer_injection'] = tmax_pi_vs_e_F
            self.stat_cache[cached_group_name + '/vmax_correlation_transfer_injection'] = vmax_pi_vs_e_F
            self.stat_cache[cached_group_name + '/correlation_dissipation_injection'] = cf_ssd_vs_e_F
            self.stat_cache[cached_group_name + '/tmax_correlation_dissipation_injection'] = tmax_ssd_vs_e_F
            self.stat_cache[cached_group_name + '/vmax_correlation_dissipation_injection'] = vmax_ssd_vs_e_F
            self.stat_cache[cached_group_name + '/correlation_dissipation_transfer'] = cf_ssd_vs_pi
            self.stat_cache[cached_group_name + '/tmax_correlation_dissipation_transfer'] = tmax_ssd_vs_pi
            self.stat_cache[cached_group_name + '/vmax_correlation_dissipation_transfer'] = vmax_ssd_vs_pi
            self.stat_cache[cached_group_name + '/correlation_dissipation_E2D'] = cf_ssd_vs_E2D
            self.stat_cache[cached_group_name + '/tmax_correlation_dissipation_E2D'] = tmax_ssd_vs_E2D
            self.stat_cache[cached_group_name + '/vmax_correlation_dissipation_E2D'] = vmax_ssd_vs_E2D
            self.stat_cache[cached_group_name + '/correlation_dissipation_Efluctuations'] = cf_ssd_vs_Efluctuations
            self.stat_cache[cached_group_name + '/tmax_correlation_dissipation_Efluctuations'] = tmax_ssd_vs_Efluctuations
            self.stat_cache[cached_group_name + '/vmax_correlation_dissipation_Efluctuations'] = vmax_ssd_vs_Efluctuations
            self.stat_cache[cached_group_name + '/correlation_Efluctuations_vs_injection'] = cf_Efluctuations_vs_e_F
            self.stat_cache[cached_group_name + '/tmax_correlation_Efluctuations_vs_injection'] = tmax_Efluctuations_vs_e_F
            self.stat_cache[cached_group_name + '/vmax_correlation_Efluctuations_vs_injection'] = vmax_Efluctuations_vs_e_F
            self.stat_cache[cached_group_name + '/correlation_E2D_vs_injection'] = cf_E2D_vs_e_F
            self.stat_cache[cached_group_name + '/tmax_correlation_E2D_vs_injection'] = tmax_E2D_vs_e_F
            self.stat_cache[cached_group_name + '/vmax_correlation_E2D_vs_injection'] = vmax_E2D_vs_e_F
            self.stat_cache[cached_group_name + '/correlation_Efluctuations_vs_E2D'] = cf_Efluctuations_vs_E2D
            self.stat_cache[cached_group_name + '/tmax_correlation_Efluctuations_vs_E2D'] = tmax_Efluctuations_vs_E2D
            self.stat_cache[cached_group_name + '/vmax_correlation_Efluctuations_vs_E2D'] = vmax_Efluctuations_vs_E2D
        data = {}
        for kk in ['t',
                   'transfer_2D_small_scales',
                   'injection_rate',
                   'E2D',
                   'Efluctuations',
                   'small_scale_dissipation',
                   'correlation_transfer_injection',
                   'tmax_correlation_transfer_injection',
                   'vmax_correlation_transfer_injection',
                   'correlation_dissipation_injection',
                   'tmax_correlation_dissipation_injection',
                   'vmax_correlation_dissipation_injection',
                   'correlation_dissipation_transfer',
                   'tmax_correlation_dissipation_transfer',
                   'vmax_correlation_dissipation_transfer',
                   'correlation_dissipation_E2D',
                   'tmax_correlation_dissipation_E2D',
                   'vmax_correlation_dissipation_E2D',
                   'correlation_dissipation_Efluctuations',
                   'tmax_correlation_dissipation_Efluctuations',
                   'vmax_correlation_dissipation_Efluctuations',
                   'correlation_Efluctuations_vs_E2D',
                   'tmax_correlation_Efluctuations_vs_E2D',
                   'vmax_correlation_Efluctuations_vs_E2D',
                   'correlation_E2D_vs_injection',
                   'tmax_correlation_E2D_vs_injection',
                   'vmax_correlation_E2D_vs_injection',
                   'correlation_Efluctuations_vs_injection',
                   'tmax_correlation_Efluctuations_vs_injection',
                   'vmax_correlation_Efluctuations_vs_injection',]:
            data[kk] = self.stat_cache[cached_group_name + '/' + kk][()]
        data['e_F'] = data['injection_rate']
        data['Pi'] = data['transfer_2D_small_scales']
        return data
    def compute_vorticity_PDF_statistics(
            self,
            iter0 = 0,
            iter1 = None,
            iter_skip = None,
            force = False):
        iterations = self.reset_iterations(iter0 = iter0, iter1 = iter1, iter_skip = iter_skip)
        assert((iter_skip % self.parameters['niter_stat']) == 0)
        cached_group_name = 'vorticity_PDF_statistics'
        if force:
            if cached_group_name in self.stat_cache.keys():
                del self.stat_cache[cached_group_name]
        if cached_group_name not in self.stat_cache.keys():
            ii = range(iterations[0] // self.parameters['niter_stat'],
                       iterations[-1]// self.parameters['niter_stat']+1,
                       iter_skip // self.parameters['niter_stat'])
            self.stat_cache[cached_group_name + '/m2'] = self.get_data_file()['statistics/moments/vorticity'][ii, 2].mean(axis = 0)
            self.stat_cache[cached_group_name + '/m4'] = self.get_data_file()['statistics/moments/vorticity'][ii, 4].mean(axis = 0)
            self.stat_cache[cached_group_name + '/m6'] = self.get_data_file()['statistics/moments/vorticity'][ii, 6].mean(axis = 0)
            self.stat_cache[cached_group_name + '/m8'] = self.get_data_file()['statistics/moments/vorticity'][ii, 8].mean(axis = 0)
            self.stat_cache[cached_group_name + '/histograms'] = self.get_data_file()['statistics/histograms/vorticity'][ii].mean(axis = 0)
            binsize = 2*(self.parameters['max_vorticity_estimate']/(3**0.5)) / self.parameters['histogram_bins']
            bincenters = np.linspace(
                    -self.parameters['max_vorticity_estimate'] + binsize/2,
                     self.parameters['max_vorticity_estimate'] - binsize/2,
                     self.parameters['histogram_bins']) / (3**0.5)
            self.stat_cache[cached_group_name + '/bincenters'] = bincenters
            self.stat_cache[cached_group_name + '/pdfs'] = self.stat_cache[cached_group_name + '/histograms'][..., :3] / (
                    self.parameters['nx']*self.parameters['ny']*self.parameters['nz']*binsize)
        data = {}
        for kk in ['m2', 'm4', 'm6', 'm8', 'pdfs', 'bincenters', 'histograms']:
            data[kk] = self.stat_cache[cached_group_name + '/' + kk][()]
        return data



if __name__ == '__main__':
    pass
