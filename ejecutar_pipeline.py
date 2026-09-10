"""Abrir en VS Code y ejecutar. Editar config_protocolo.py para elegir entradas."""
from pathlib import Path
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import traceback
from uuid import uuid4
import warnings

import mne
import numpy as np
import config_protocolo as cfg
from config_sujetos import SUBJECT_CONFIG
from config_regiones import REGIONS
import features as fx
from segmentacion import extract_events, build_intervals, trial_info
from salidas import write_json, jsonable, sha256_file, save_details, quality_rows, export_workbook, ARRAY_AXES, contribution_rows

ALL_FEATURES = ['spectral','wpli','wsmi','pe','lzc','te']


def versions():
    result={'python':sys.version,'executable':sys.executable}
    for name in ['mne','mne-connectivity','numpy','scipy','scikit-learn','specparam','pandas','openpyxl']:
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    return result


def configuration():
    return {key:getattr(cfg,key) for key in dir(cfg) if key.isupper()}


def clean_subject(subject, settings, version_info):
    paths=[cfg.DATA_DIR/(subject+extension) for extension in ['.bdf','.fif']]
    source=next((p for p in paths if p.is_file()),None)
    if source is None: raise FileNotFoundError(f'No se encontro {subject}.bdf/.fif en {cfg.DATA_DIR}')
    fingerprint=dict(input=str(source.resolve()),input_sha256=sha256_file(source),
        preprocessing=cfg.PREPROCESS,ica=cfg.ICA_SETTINGS,subject_settings=settings,
        versions={k:version_info[k] for k in ['mne','numpy','scipy','scikit-learn']},
        code=inspect.getsource(fx.preprocessing_mne)+inspect.getsource(fx.make_ICA))
    digest=hashlib.sha256(json.dumps(jsonable(fingerprint),sort_keys=True).encode()).hexdigest()
    folder=cfg.CACHE_DIR/subject/digest[:20]
    raw_path=folder/'clean_raw.fif'; ica_path=folder/'solution-ica.fif'; manifest=folder/'cache.json'
    if cfg.REUSE_CLEAN_CACHE and manifest.exists():
        stored=json.loads(manifest.read_text(encoding='utf-8'))
        if (stored.get('fingerprint')==jsonable(fingerprint) and raw_path.is_file() and ica_path.is_file()
            and stored.get('raw_sha256')==sha256_file(raw_path) and stored.get('ica_sha256')==sha256_file(ica_path)):
            print('Reutilizando limpieza e ICA:',raw_path)
            return mne.io.read_raw_fif(raw_path,preload=True,verbose=False),dict(source=str(source),
                cache=str(folder),cache_reused=True,input_sha256=fingerprint['input_sha256'])
    folder.mkdir(parents=True,exist_ok=True)
    # Marcadores originales intactos; ambas representaciones usan este unico EEG limpio.
    raw,_=fx.preprocessing_mne(path=str(cfg.DATA_DIR),file=subject,bads=settings['bad_channels'],
        raw_plot=False,filtered_plot=False,psd_plot=False,edit_marks=False,**cfg.PREPROCESS)
    ica,clean=fx.make_ICA(raw,bad_ica_channels=settings['ica_exclude'],plot_ica_topo=False,
                        plot_ica_time=False,plot_raw=False,**cfg.ICA_SETTINGS)
    if not np.array_equal(raw.get_data(picks=[cfg.STATUS_CHANNEL]),clean.get_data(picks=[cfg.STATUS_CHANNEL])):
        raise ValueError('ICA modifico el canal de eventos.')
    # Cache publicada solo cuando todos sus componentes terminaron de guardarse.
    ica.save(ica_path,overwrite=True)
    clean.save(raw_path,fmt='double',overwrite=True,verbose=False)
    write_json(manifest,dict(fingerprint=fingerprint,raw_sha256=sha256_file(raw_path),ica_sha256=sha256_file(ica_path)))
    return clean,dict(source=str(source),cache=str(folder),cache_reused=False,input_sha256=fingerprint['input_sha256'])


def region_report(raw, subject):
    picks=mne.pick_types(raw.info,eeg=True,meg=False,stim=False,exclude=[])
    available=[raw.ch_names[i] for i in picks]
    rows=[]
    for name,channels in REGIONS.items():
        if len(channels)!=len(set(channels)): raise ValueError(f'Canales duplicados en {name}.')
        used=[ch for ch in channels if ch in available]
        missing=[ch for ch in channels if ch not in available]
        rows.append(dict(subject=subject,region=name,used_channels=used,missing_channels=missing,
                         n_used=len(used),interpolated_channels=[c for c in channels
                         if c in SUBJECT_CONFIG[subject]['bad_channels'] and cfg.PREPROCESS['interpolate']]))
        if missing: print(f'AVISO: {name}: faltan {missing}')
        if not used: raise ValueError(f'Region {name} sin canales disponibles.')
    # wSMI asigna cada canal a una region; impedir membresias ambiguas.
    flat=[c for chs in REGIONS.values() for c in chs]
    if len(flat)!=len(set(flat)): raise ValueError('Regiones superpuestas: wSMI requiere asignacion unica.')
    return rows


def calculate(name, raw, band, condition, windows, blocks):
    common=dict(status_channel=cfg.STATUS_CHANNEL,status_start_code=cfg.PROTOCOL[condition]['start_code'],
                status_end_code=-1,trial_mode='average')
    win=trial_info(windows,raw.info['sfreq'])
    if name=='wpli':
        step=round(cfg.WPLI_EPOCH_SECONDS*raw.info['sfreq'])
        if step<=0: raise ValueError('Duracion de epoca wPLI invalida.')
        for block in blocks:
            n,remain=divmod(block['stop']-block['start'],step)
            if remain or n<cfg.WPLI_MIN_EPOCHS:
                raise ValueError(f'wPLI: bloque {block["block"]}: {n} epocas completas y {remain} muestras restantes.')
        return fx.phase_connectivity_wpli(raw,band_range=band,trial_info=trial_info(blocks,raw.info['sfreq']),
                                          epoch_seconds=cfg.WPLI_EPOCH_SECONDS,**common)
    common['trial_info']=win
    if name=='spectral': return fx.spectral_parametrization(raw,band_range=band,**common,**cfg.SPECTRAL_SETTINGS)
    if name=='wsmi': return fx.patterns_connectivity_wsmi(raw,band_range=band,**common,**cfg.WSMI_SETTINGS)
    if name=='pe': return fx.permutation_entropy(raw,band_range=band,**common,**cfg.PE_SETTINGS)
    if name=='lzc': return fx.lempel_ziv_complexity(raw,**common,**cfg.LZC_SETTINGS)
    if name=='te': return fx.transfer_entropy(raw,**common,**cfg.TE_SETTINGS)
    raise ValueError(name)


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def main():
    selected=cfg.SUBJECTS_TO_RUN if cfg.SUBJECTS_TO_RUN is not None else list(SUBJECT_CONFIG)
    if not selected or len(selected)!=len(set(selected)): raise ValueError('Seleccion de sujetos vacia/duplicada.')
    if not cfg.FEATURES_TO_RUN or set(cfg.FEATURES_TO_RUN)-set(ALL_FEATURES): raise ValueError('Features invalidas.')
    if len(cfg.FEATURES_TO_RUN)!=len(set(cfg.FEATURES_TO_RUN)): raise ValueError('Features repetidas.')
    for subject in selected:
        if subject not in SUBJECT_CONFIG: raise ValueError(f'Falta configuracion del sujeto {subject}.')
    for lo,hi in cfg.BANDS.values():
        if not 0<lo<hi: raise ValueError('Limites de banda invalidos.')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    folder=cfg.OUTPUT_DIR/run_id; folder.mkdir(parents=True,exist_ok=False)
    code_dir=folder/'codigo'; code_dir.mkdir()
    for source in Path(__file__).parent.glob('*.py'): shutil.copy2(source,code_dir/source.name)
    info=dict(run_id=run_id,versions=versions(),configuration=configuration(),subjects=SUBJECT_CONFIG,
              regions=REGIONS,script_sha256={p.name:sha256_file(p) for p in code_dir.glob('*.py')},
              status='running',definitions={
                'spec_period':'Integral PSD/fondo aperiodico, escala lineal; NO potencia absoluta.',
                'wpli':'MNE Fourier; por bloque con epocas conjuntas; media entre bloques si existen varios.',
                'lzc':'Banda ancha; repetida en filas de bandas para compatibilidad.',
                'te':'Banda ancha original; repetida en filas de bandas para compatibilidad.'})
    write_json(folder/'configuracion.json',info)
    rows=[]; trace=[]; all_intervals=[]; issues=[]; quality=[]; regions=[]
    output=folder/'EEG_features_subject_level.xlsx'
    def flush(): export_workbook(output,rows,trace,all_intervals,issues,quality,regions,cfg.EXPORT_CSV)
    with open(folder/'ejecucion.log','w',encoding='utf-8') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            for subject in selected:
                print('\nSUJETO:',subject)
                try:
                    raw,origin=clean_subject(subject,SUBJECT_CONFIG[subject],info['versions'])
                    regions.extend(region_report(raw,subject))
                    events=extract_events(raw,cfg.STATUS_CHANNEL,cfg.STATUS_MASK)
                    blocks,windows,problems=build_intervals(events,raw.n_times,raw.info['sfreq'],cfg.PROTOCOL,
                        cfg.WINDOW_SECONDS,cfg.MARK_TOLERANCE_SECONDS,cfg.REMAINDER_POLICY)
                    write_json(folder/'eventos'/f'{subject}.json',dict(origin=origin,sfreq=raw.info['sfreq'],
                        first_samp=raw.first_samp,events=events,blocks=blocks,windows=windows,issues=problems))
                    issues.extend(dict(subject=subject,**p) for p in problems)
                    all_intervals.extend(dict(subject=subject,level=level,**i) for level,ints in [('block',blocks),('window',windows)] for i in ints)
                except Exception as error:
                    issues.append(dict(subject=subject,stage='preprocessing',status='failed',message=str(error)))
                    traceback.print_exc(); flush(); continue
                for condition in cfg.PROTOCOL:
                    cond_windows=[w for w in windows if w['condition']==condition]
                    cond_blocks=[b for b in blocks if b['condition']==condition]
                    if not cond_blocks: continue
                    for band_name,band in cfg.BANDS.items():
                        results={}
                        for name in cfg.FEATURES_TO_RUN:
                            key=dict(subject=subject,condition=condition,band=band_name,feature=name)
                            print('CALCULO:',key)
                            metadata=dict(**key,run_id=run_id,origin=origin,sfreq=raw.info['sfreq'],
                                window_intervals=cond_windows,block_intervals=cond_blocks,
                                analysis_scope='block_joint_subepochs' if name=='wpli' else 'individual_windows',
                                array_axes=ARRAY_AXES,requested_band=band,parameters=(dict(method='wpli',mode='fourier',faverage=True,
                                    epoch_seconds=cfg.WPLI_EPOCH_SECONDS) if name=='wpli' else getattr(cfg,{'spectral':'SPECTRAL_SETTINGS',
                                    'pe':'PE_SETTINGS','wsmi':'WSMI_SETTINGS','lzc':'LZC_SETTINGS','te':'TE_SETTINGS'}[name])),
                                versions=info['versions'],eeg_channels=[raw.ch_names[i] for i in mne.pick_types(raw.info,eeg=True,exclude=[])])
                            try:
                                with warnings.catch_warnings(record=True) as captured:
                                    warnings.simplefilter('always')
                                    result=calculate(name,raw,band,condition,cond_windows,cond_blocks)
                                for warning in captured:
                                    issues.append(dict(**key,stage='calculation',status='warning',message=str(warning.message)))
                                base=folder/'detalles'/subject/f'{condition}_{band_name}_{name}'
                                save_details(base,result,metadata)
                                quality.extend(quality_rows(result,key))
                                contributors=contribution_rows(result,key)
                                quality.extend(contributors)
                                incomplete=any(q['n_finite']<q['n_values'] for q in contributors)
                                if incomplete:
                                    issues.append(dict(**key,stage='quality',status='partial',message='Hay agregados con contribuciones no finitas; revisar calidad.'))
                                results[name]=result
                                counts=result.get('epoch_counts',[]) if name=='wpli' else [len(cond_windows)]
                                trace.append(dict(**key,run_id=run_id,status='partial' if incomplete else 'computed',n_windows=len(cond_windows),
                                    n_blocks=len(cond_blocks),duration_s=sum(b['duration_s'] for b in cond_blocks),
                                    epoch_counts=counts,details=str(base.relative_to(folder)),cache_reused=origin['cache_reused']))
                                for error in result.get('errors',[]):
                                    issues.append(dict(**key,stage='estimator',status='partial',message=json.dumps(jsonable(error))))
                            except Exception as error:
                                issues.append(dict(**key,stage='feature',status='failed',message=str(error)))
                                trace.append(dict(**key,run_id=run_id,status='failed',message=str(error)))
                                traceback.print_exc()
                        # Una fila contiene unicamente las features calculadas en esta corrida.
                        if results:
                            row=fx._build_aggregated_row(subject=subject,band=band_name,condition=condition,
                                **{dict(spectral='spectral_results',wpli='wpli_results',wsmi='wsmi_results',pe='pe_results',
                                        lzc='lzc_results',te='te_results')[k]:v for k,v in results.items()})
                            rows.append(row)
                        flush()  # Fallos de exportacion detienen la corrida; no se ocultan.
                del raw
            info['status']='completed_with_issues' if issues else 'completed'
            flush()
        except Exception:
            info['status']='aborted'; raise
        finally:
            write_json(folder/'configuracion.json',info)
            write_json(folder/'incidencias.json',issues)
    print('\nResultados:',folder)
    print('Estado:',info['status'])
    return folder


if __name__=='__main__':
    main()
