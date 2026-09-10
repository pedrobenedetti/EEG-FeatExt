"""Eventos originales e intervalos [start, stop), relativos al Raw (no first_samp)."""
import numpy as np


def extract_events(raw, channel="Status", mask=None):
    values = np.rint(raw.get_data(picks=[channel])[0]).astype(np.int64)
    if mask is not None:
        values = values & int(mask)
    previous = np.r_[0, values[:-1]]
    positions = np.flatnonzero((values != 0) & (values != previous))
    return [dict(sample=int(i), code=int(values[i]), seconds=float(i / raw.info['sfreq']))
            for i in positions]


def build_intervals(events, n_times, sfreq, protocol, window_seconds=5,
                    tolerance_seconds=0.1, remainder_policy="error"):
    """Una inconsistencia invalida esa condicion; nunca cambia duraciones en silencio."""
    blocks, windows, issues = [], [], []
    starts_all = {int(c['start_code']) for c in protocol.values()}
    if window_seconds <= 0 or tolerance_seconds < 0 or remainder_policy not in {'error', 'drop'}:
        raise ValueError('Configuracion de ventanas/tolerancia invalida.')
    for condition, rule in protocol.items():
        local_blocks, local_windows = [], []
        try:
            mode = rule['mode']
            candidates = [e for e in events if e['code'] == rule['start_code']]
            if not candidates:
                raise ValueError('No se encontro la marca inicial.')
            consumed_until = -1
            for event in candidates:
                anchor = event['sample']
                if mode == 'fixed' and anchor < consumed_until:
                    continue
                if mode == 'fixed':
                    start = anchor
                    stop = start + int(round(rule['duration_s'] * sfreq))
                    offsets = rule.get('repeated_offsets_s', [])
                    if any(o <= 0 or o >= rule['duration_s'] for o in offsets):
                        raise ValueError('Offsets intermedios fuera del bloque.')
                    inside = [e for e in candidates if start < e['sample'] < stop]
                    expected = [start + round(float(o) * sfreq) for o in sorted(offsets)]
                    if len(inside) != len(expected):
                        raise ValueError(f'Marcas intermedias: {len(inside)}; esperadas: {len(expected)}.')
                    deviations = [(e['sample'] - target) / sfreq for e, target in zip(inside, expected)]
                    if any(abs(d) > tolerance_seconds for d in deviations):
                        raise ValueError(f'Desviaciones de marcas (s): {deviations}.')
                    foreign = [e for e in events if start < e['sample'] < stop
                               and e['code'] in starts_all and e['code'] != rule['start_code']]
                    if foreign:
                        raise ValueError('Otra condicion comienza dentro del bloque previsto.')
                    consumed_until = stop
                elif mode == 'until_marker':
                    start = anchor
                    endings = [e for e in events if e['sample'] > start and e['code'] == rule['end_code']]
                    if not endings:
                        raise ValueError('Falta marca de cierre.')
                    stop = endings[0]['sample']
                    if any(start < e['sample'] < stop for e in candidates):
                        raise ValueError('Nuevo inicio antes de cerrar el trial anterior.')
                    duration = (stop-start)/sfreq
                    if not rule.get('min_duration_s', 0) <= duration <= rule.get('max_duration_s', float('inf')):
                        raise ValueError(f'Duracion fuera de rango: {duration} s.')
                    deviations = []
                elif mode == 'event':
                    start = anchor + int(round(rule['tmin_s'] * sfreq))
                    stop = anchor + int(round(rule['tmax_s'] * sfreq))
                    deviations = []
                else:
                    raise ValueError(f'Modo desconocido: {mode}')
                if start < 0 or stop > n_times or stop <= start:
                    raise ValueError(f'Intervalo [{start}, {stop}) fuera de la grabacion o vacio.')
                block_id = len(local_blocks) + 1
                step = stop-start if mode == 'event' else int(round(window_seconds*sfreq))
                if step < 1:
                    raise ValueError('Ventana menor a una muestra.')
                count, remainder = divmod(stop-start, step)
                if count == 0 or (remainder and remainder_policy == 'error'):
                    raise ValueError('El bloque no se divide exactamente en ventanas completas.')
                block = dict(condition=condition, condition_name=rule['name'], block=block_id,
                             start=start, stop=stop, anchor=anchor, duration_s=(stop-start)/sfreq,
                             mode=mode, marker_deviations_s=deviations,
                             window_remainder_samples=remainder)
                local_blocks.append(block)
                for number in range(count):
                    local_windows.append(dict(condition=condition, condition_name=rule['name'], block=block_id,
                        window_in_block=number+1, window_in_condition=len(local_windows)+1,
                        start=start+number*step, stop=start+(number+1)*step, duration_s=step/sfreq))
            expected_blocks = rule.get('expected_blocks')
            if expected_blocks is not None and len(local_blocks) != expected_blocks:
                raise ValueError(f'{len(local_blocks)} bloques; se esperaban {expected_blocks}.')
            blocks.extend(local_blocks); windows.extend(local_windows)
        except (ValueError, KeyError, TypeError) as error:
            issues.append(dict(condition=condition, stage='segmentation', status='invalid', message=str(error)))
    # Solapamientos invalidan ambas condiciones (evita duplicar muestras sin declararlo).
    invalid = set()
    ordered = sorted(blocks, key=lambda b:b['start'])
    for i, block in enumerate(ordered):
        for other in ordered[i+1:]:
            if other['start'] >= block['stop']:
                break
            invalid.update([block['condition'], other['condition']])
    for condition in invalid:
        issues.append(dict(condition=condition, stage='segmentation', status='invalid',
                           message='Bloques solapados: se requiere una politica explicita para este protocolo.'))
    return ([b for b in blocks if b['condition'] not in invalid],
            [w for w in windows if w['condition'] not in invalid], issues)


def trial_info(intervals, sfreq):
    if not intervals:
        raise ValueError('No hay intervalos validos.')
    return dict(trial_indices=list(range(len(intervals))),
                trial_starts=[i['start'] for i in intervals], trial_ends=[i['stop'] for i in intervals],
                trial_values=[i['condition'] for i in intervals],
                trial_end_values=[-1]*len(intervals), n_trials=len(intervals), sfreq=sfreq)
