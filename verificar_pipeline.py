"""Pruebas sinteticas. Ejecutar desde VS Code; no lee ni modifica tus EEG."""
from pathlib import Path
import contextlib
import io
import json
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs
import features as fx
import config_protocolo as cfg
from config_regiones import REGIONS
from segmentacion import extract_events, build_intervals, trial_info
from salidas import save_details, export_workbook, contribution_rows


def synthetic(seconds=11, seed=12):
    sf=500; names=[ch for channels in REGIONS.values() for ch in channels[:2]]
    n=int(seconds*sf); rng=np.random.default_rng(seed); t=np.arange(n)/sf
    x=np.stack([1e-6*(np.sin(2*np.pi*10*t+i*.3)+rng.normal(0,.7,n)) for i in range(len(names))])
    raw=mne.io.RawArray(np.vstack([x,np.zeros(n)]),mne.create_info(names+['Status'],sf,['eeg']*len(names)+['stim']),verbose=False)
    return raw


class Tests(unittest.TestCase):
    def test_01_default_protocol(self):
        raw=synthetic(181)
        for i,code in enumerate([40,40,60,60,100,100]): raw._data[-1,1+i*15000]=code
        events=extract_events(raw)
        b,w,errors=build_intervals(events,raw.n_times,500,cfg.PROTOCOL)
        self.assertEqual(errors,[]); self.assertEqual(len(b),3); self.assertEqual(len(w),36)
        for code in cfg.PROTOCOL:
            self.assertEqual(len([i for i in w if i['condition']==code]),12)
        missing=[e for e in events if e['sample']!=15001]
        b,w,errors=build_intervals(missing,raw.n_times,500,cfg.PROTOCOL)
        self.assertEqual(len(errors),1); self.assertFalse(any(i['condition']==40 for i in w))
        moved=[dict(e,sample=e['sample']+51) if e['sample']==15001 else e for e in events]
        self.assertEqual(len(build_intervals(moved,raw.n_times,500,cfg.PROTOCOL)[2]),1)
        self.assertTrue(build_intervals(events,89000,500,cfg.PROTOCOL)[2])
        raw._data[-1,0]=140
        self.assertEqual(extract_events(raw)[0]['sample'],0)

    def test_02_other_protocols(self):
        events=[dict(sample=1000,code=10),dict(sample=6000,code=20)]
        p={1:dict(name='a',mode='until_marker',start_code=10,end_code=20)}
        b,w,e=build_intervals(events,10000,500,p)
        self.assertEqual((len(b),len(w),len(e)),(1,2,0))
        p={1:dict(name='a',mode='event',start_code=10,tmin_s=-1,tmax_s=3)}
        b,w,e=build_intervals(events,10000,500,p)
        self.assertEqual((w[0]['start'],w[0]['stop']),(500,2500))
        p={1:dict(name='a',mode='fixed',start_code=10,duration_s=11)}
        self.assertTrue(build_intervals(events,10000,500,p)[2])
        b,w,e=build_intervals(events,10000,500,p,remainder_policy='drop')
        self.assertEqual(b[0]['window_remainder_samples'],500)
        self.assertEqual(len(w),2)

    def test_03_wpli_joint_and_matrix(self):
        raw=synthetic(61)
        intervals=[dict(start=1,stop=30001,condition=40)]
        result=fx.phase_connectivity_wpli(raw,band_range=(8,12),trial_mode='average',status_channel='Status',status_start_code=40,
                                         trial_info=trial_info(intervals,500))
        self.assertEqual(result['epoch_counts'],[12])
        matrix=result['wpli_channels'][0]
        filtered=raw.copy().filter(8,12,picks='eeg',method='iir',verbose=False)
        data=filtered.get_data(picks='eeg',start=1,stop=30001)
        epochs=data.reshape(len(data),12,2500).transpose(1,0,2)
        ii,jj=np.tril_indices(len(data),-1)
        conn=spectral_connectivity_epochs(epochs,sfreq=500,indices=(ii,jj),method='wpli',
            mode='fourier',fmin=8,fmax=12,faverage=True,verbose=False)
        np.testing.assert_allclose(matrix[ii,jj],conn.get_data()[:,0],atol=1e-12)
        np.testing.assert_allclose(matrix,matrix.T)
        self.assertTrue(np.all(np.diag(matrix)==0)); self.assertTrue(np.any(matrix[ii,jj]>0))
        for i,name in enumerate(result['zone_names']):
            indices=[result['channel_names'].index(ch) for ch in REGIONS[name][:2]]
            self.assertAlmostEqual(result['wpli_trials'][0,i,i],matrix[indices[0],indices[1]])

    def test_04_original_features_regression(self):
        original=runpy.run_path(str(Path(__file__).parent/'referencia'/'featureExtraction_original.py'),run_name='reference')
        raw=synthetic()
        # Mismas dos ventanas en ambos caminos: original por marcas; nuevo por intervalos.
        raw._data[-1,1]=40; raw._data[-1,2501]=5; raw._data[-1,2503]=40; raw._data[-1,5003]=5
        info=trial_info([dict(start=1,stop=2501,condition=40),dict(start=2503,stop=5003,condition=40)],500)
        common=dict(status_channel='Status',status_start_code=40,status_end_code=5,trial_mode='average')
        methods=[('spectral_parametrization',dict(band_range=(8,12),freq_range=(1,30)),['theta_power_zones','aperiodic_exponent_zones','aperiodic_offset_zones']),
                 ('patterns_connectivity_wsmi',dict(band_range=(8,12),debug_first_pair=False),['wsmi_matrix']),
                 ('permutation_entropy',dict(band_range=(8,12)),['pe_matrix_channels','pe_matrix_zones']),
                 ('lempel_ziv_complexity',{},['lzc_matrix_channels','lzc_matrix_zones']),
                 ('transfer_entropy',dict(maxlag_ms=10),['te_full','te_mean_lag'])]
        for name,kwargs,keys in methods:
            with self.subTest(feature=name):
                old=original[name](raw,**common,**kwargs)
                new=getattr(fx,name)(raw,trial_info=info,**common,**kwargs)
                for key in keys:
                    self.assertTrue(np.isfinite(new[key]).any(),key)
                    np.testing.assert_allclose(new[key],old[key],rtol=1e-12,atol=1e-12,equal_nan=True)

    def test_05_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); result={'pe_matrix_zones':np.array([[.1,np.nan],[.2,.3]]),'zone_names':['a','b']}
            save_details(p/'detail',result,{'subject':'001'})
            with np.load(p/'detail.npz',allow_pickle=False) as arrays: self.assertEqual(arrays['result__pe_matrix_zones'].shape,(2,2))
            counts=contribution_rows(result,{})
            self.assertEqual([r['n_finite'] for r in counts],[1,2])
            rows=[dict(subject='001',band='alpha',condition=40,pe_a=.123456789123456)]
            target=p/'results.xlsx'
            export_workbook(target,rows,[],[],[],[],[],csv=True)
            data=pd.read_excel(target,sheet_name='subject_level',dtype={'subject':str})
            self.assertEqual(data['subject'][0],'001'); self.assertAlmostEqual(data['pe_a'][0],rows[0]['pe_a'],places=14)
            before=target.read_bytes()
            with patch('salidas.os.replace',side_effect=PermissionError('archivo abierto')):
                with self.assertRaises(PermissionError): export_workbook(target,rows,[],[],[],[],[])
            self.assertEqual(before,target.read_bytes())
            with self.assertRaises(ValueError): export_workbook(target,rows*2,[],[],[],[],[])


if __name__=='__main__':
    unittest.main(argv=['verificar_pipeline'],verbosity=2)
