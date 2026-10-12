"""Verify and explain preserved aggregate evidence without SUMO or model fitting."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    """Load a bundled JSON file relative to this checkout."""
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def verify_exports():
    """Verify source/evidence exports, independent of Windows/Linux newlines."""
    manifest = read('PROVENANCE.json')
    for entry in manifest['exports']:
        content = (ROOT / entry['portfolio_path']).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f'Export hash mismatch: {entry["portfolio_path"]}')
    return manifest


def reductions(totals, package, reference='portfolio'):
    """Recompute percent reductions; a negative result denotes a regression."""
    return {metric: 100 * (baseline - totals[package][metric]) / baseline
            for metric, baseline in totals[reference].items() if baseline > 0}


def build_report():
    """Generate a case-study report from included evidence, with regressions visible."""
    manifest = verify_exports()
    controllers = read('evidence/controllers.json')
    recon = read('evidence/reconstruction.json')
    forward = read('evidence/forward.json')
    rows = ['# Preserved traffic research results', '',
            f'Main source revision: `{manifest["main_commit"]}`.', '',
            'Historical measurements, not a new simulation or a rebenchmark of reviewed code.', '',
            '## Controller comparison', '',
            'Positive values mean lower aggregate cost; negative values mean regression.', '',
            '| Set/profile | Stops | Waiting | Time loss | CO2 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for setting, packages in [('primary', ['mlp', 'neural_ranker']), ('alternate', ['mlp'])]:
        data = controllers[setting]
        for package in packages:
            result = reductions(data['totals'], package)
            rows.append('| '+setting+'/'+package+' | '+' | '.join(
                f'{result[k]:+.2f}%' for k in ['stops_created', 'waiting_time_s', 'time_loss_s', 'co2_mg'])+' |')
    rows += ['', 'The stop-first ranker reduces stops while increasing waiting. A separate route set reverses some MLP gains.', '',
             '## Reconstruction acceptance', '',
             '| Replay | Boundary timing MAE | Internal matches/raw | Internal timing MAE | Accepted |',
             '| --- | ---: | ---: | ---: | --- |']
    for name in ['accepted', 'trial']:
        item = recon[name]; boundary = item['groups']['boundary']; internal = item['groups']['internal']
        rows.append(f'| {name} | {boundary["mae_s"]:.6f} s | {internal["matched"]}/{internal["raw"]} | {internal["mae_s"]:.4f} s | {item["accepted_constraints"]} |')
    rows += ['', 'Anonymous event matches are not persistent real-vehicle identities. Internal MAE is conditional on matched observations; unmatched penalties and coverage must also be considered.', '',
             '## Conditional forward surrogate', '',
             f'Exact sequence accuracy: {100 * forward["model"]["exact_sequence_accuracy"]:.2f}%.',
             f'First/second detector timing MAE: {forward["model"]["first_detection_time_mae_s"]:.3f} / {forward["model"]["second_detection_time_mae_s"]:.3f} s.', '',
             'Vehicle identities are held out as groups. Surrounding traffic history is supplied, including future context. This is conditional surrogate evaluation, not independent-day validation or a deployable inverse model.']
    return '\n'.join(rows)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / '.demo')
    args = parser.parse_args()
    report = build_report()
    if args.check:
        print('Export integrity and saved-result report: OK')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'REPORT.md').write_text(report, encoding='utf-8')
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Traffic research evidence</title><style>body{max-width:1000px;margin:3rem auto;padding:0 1.5rem;background:#f7f4eb;color:#17352f;font:16px/1.6 Georgia,serif}pre{white-space:pre-wrap;font:14px/1.6 monospace}</style><h1>Traffic research evidence</h1><pre>'+html.escape(report)+'</pre></html>'
    (args.out / 'report.html').write_text(body, encoding='utf-8')
    print(f'Wrote {args.out / "REPORT.md"} and report.html')


if __name__ == '__main__':
    main()
