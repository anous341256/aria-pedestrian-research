"""Losslessly package original CVAT RLE masks for append-only CVAT import."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/sam2_fullvideos_20260927'
DEST = BASE / 'cvat_masks'
LABEL = {'name': 'person_sam2_mask', 'color': '#ffcc00', 'type': 'mask',
         'attributes': [{'name': 'sam2_id', 'input_type': 'text', 'mutable': False,
                         'values': ['unknown']}]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    rows = []
    staging = json.loads((BASE / 'cvat_upload/staging_manifest.json').read_text())
    for row in staging:
        sid = row['recording']
        source = BASE / sid / 'sam2_raw_v1/masks_rle.jsonl.gz'
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = '1.1'
        meta = copy.deepcopy(ET.parse(row['xml']).find('meta'))
        root.append(meta)
        label = meta.find('task/labels/label')
        label.find('name').text = LABEL['name']
        if label.find('type') is None:
            ET.SubElement(label, 'type').text = 'mask'
        else:
            label.find('type').text = 'mask'
        tracks = {i: ET.SubElement(root, 'track', {'id': str(i), 'label': LABEL['name'],
                  'source': 'auto', 'group_id': str(i)}) for i in range(1, 6)}
        prior_visible = {i: False for i in tracks}
        expected = {}
        nonempty = {i: 0 for i in tracks}
        all_records = 0
        with gzip.open(source, 'rt', encoding='utf-8') as handle:
            for line in handle:
                rec = json.loads(line)
                f, oid, rle = rec['frame'], rec['id'], rec['rle']
                all_records += 1
                runs, (left, top, right, bottom) = rle[:-4], rle[-4:]
                assert all(isinstance(x, int) and x >= 0 for x in runs)
                assert sum(runs) == (right - left + 1) * (bottom - top + 1)
                visible = sum(runs[1::2]) > 0
                if visible or prior_visible[oid]:
                    el = ET.SubElement(tracks[oid], 'mask', {
                        'frame': str(f), 'outside': str(int(not visible)), 'occluded': '0',
                        'keyframe': '1', 'z_order': str(oid),
                        'rle': ', '.join(map(str, runs)), 'left': str(left), 'top': str(top),
                        'width': str(right - left + 1), 'height': str(bottom - top + 1)})
                    ET.SubElement(el, 'attribute', {'name': 'sam2_id'}).text = f'S{oid}'
                    expected[(oid, f)] = rle
                nonempty[oid] += int(visible)
                prior_visible[oid] = visible
        path = DEST / f'Aria_{sid}_SAM2_original_masks.xml'
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        # Verify every imported RLE, its exact crop coordinates and nonempty count.
        parsed = ET.parse(path)
        restored = {}
        counts = {i: 0 for i in tracks}
        for track in parsed.findall('track'):
            oid = int(track.attrib['id'])
            for mask in track.findall('mask'):
                a = mask.attrib
                l, t, w, h = (int(a[k]) for k in ('left', 'top', 'width', 'height'))
                restored[(oid, int(a['frame']))] = list(map(int, a['rle'].split(','))) + [l, t, l+w-1, t+h-1]
                counts[oid] += int(a['outside'] == '0')
        assert restored == expected
        assert counts == nonempty and sum(counts.values()) == row['nonempty_boxes']
        rows.append({'recording': sid, 'xml': str(path), 'source_rle': str(source),
                     'frames': row['frames'], 'tracks': 5, 'nonempty_masks': sum(counts.values()),
                     'per_id_nonempty_masks': nonempty, 'original_records': all_records,
                     'source_rle_sha256': digest(source), 'xml_sha256': digest(path),
                     'all_imported_rle_and_crop_coordinates_exact': True,
                     'empty_runs': 'outside marker on first empty frame after visible; remains outside until next visible mask',
                     'convert_masks_to_polygons': False, 'import_mode': 'append'})
    (DEST / 'mask_label.json').write_text(json.dumps(LABEL, indent=2), encoding='utf-8')
    (DEST / 'staging_manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps([{'recording': r['recording'], 'masks': r['nonempty_masks'],
                       'xml_mb': round(Path(r['xml']).stat().st_size / 1e6, 2)} for r in rows]))


if __name__ == '__main__':
    main()
