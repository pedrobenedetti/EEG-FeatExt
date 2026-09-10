"""Prueba completa con FIF sintetico y carpetas temporales. No usa tus EEG."""
from pathlib import Path
import tempfile, json
import pandas as pd
import numpy as np
import mne
from unittest.mock import patch
from verificar_pipeline import synthetic
import ejecutar_pipeline as run
from segmentacion import extract_events


def main():
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp); cfg=run.cfg
        raw=synthetic(12)
        # Dos referencias externas, como en los datos reales.
        ref=mne.io.RawArray(np.zeros((2,raw.n_times)),mne.create_info(['EXG1','EXG2'],500,['misc','misc']),verbose=False)
        raw.add_channels([ref]); raw._data[raw.ch_names.index('Status'),1]=40
        raw.save(p/'sample.fif',fmt='double',overwrite=True,verbose=False)
        cfg.DATA_DIR=p;cfg.CACHE_DIR=p/'cache';cfg.OUTPUT_DIR=p/'output';cfg.SUBJECTS_TO_RUN=['sample']
        cfg.PROTOCOL={40:dict(name='test',mode='fixed',start_code=40,duration_s=10,repeated_offsets_s=[],expected_blocks=1)}
        cfg.BANDS={'alpha':(8,12)}; cfg.ICA_SETTINGS.update(n_components=3,decim=3)
        cfg.TE_SETTINGS['maxlag_ms']=10;cfg.FEATURES_TO_RUN=run.ALL_FEATURES
        run.SUBJECT_CONFIG['sample']={'bad_channels':[],'ica_exclude':[0]}
        first=run.main()
        info=json.loads((first/'configuracion.json').read_text()); print('FIRST',info['status'])
        table=pd.read_excel(first/'EEG_features_subject_level.xlsx',sheet_name='subject_level')
        trace=pd.read_excel(first/'EEG_features_subject_level.xlsx',sheet_name='trazabilidad')
        assert len(table)==1 and len(trace)==6
        assert set(trace.status)<= {'computed','partial'},trace.to_string()
        for prefix in ['spec_period_','wpli_','wsmi_','pe_','lzc_','te_']:
            assert any(c.startswith(prefix) for c in table),prefix
        cfg.FEATURES_TO_RUN=['wpli']
        with patch.object(run.fx.ICA,'fit',side_effect=AssertionError('No se debe recalcular ICA')):
            second=run.main()
        trace2=pd.read_excel(second/'EEG_features_subject_level.xlsx',sheet_name='trazabilidad')
        assert trace2.cache_reused.all()
        print('INTEGRATION_OK all six features, cache reuse, workbook and detail exports')


if __name__ == "__main__":
    main()
