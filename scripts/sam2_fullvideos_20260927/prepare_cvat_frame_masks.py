"""Export pixel masks as per-frame CVAT shapes (CVAT does not support mask tracks).

The initial 164557 trial imported five prompt-frame masks only. Its remainder
package skips exactly these masks, so append does not create duplicates.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/sam2_fullvideos_20260927'
DEST = BASE / 'cvat_masks'


def main():
    records = []
    for sid in ['162538', '164557', '151153', '193709', '201129']:
        old = ET.parse(DEST / f'Aria_{sid}_SAM2_original_masks.xml')
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = '1.1'
        root.append(copy.deepcopy(old.find('meta')))
        images, expected, masks = {}, {}, []
        for track in old.findall('track'):
            oid = int(track.attrib['id'])
            for mask in track.findall('mask'):
                if mask.attrib['outside'] == '0':
                    masks.append((int(mask.attrib['frame']), oid, mask))
        for f, oid, original in sorted(masks, key=lambda item: item[:2]):
            if f not in images:
                images[f] = ET.SubElement(root, 'image', {'id': str(f),
                    'name': f'frame_{f:06d}.png', 'width': '2560', 'height': '1920'})
            mask = copy.deepcopy(original)
            for key in ('frame', 'outside', 'keyframe'):
                del mask.attrib[key]
            mask.attrib.update(label='person_sam2_mask', source='auto', group_id=str(oid))
            images[f].append(mask)
            a = mask.attrib
            l, t, w, h = (int(a[k]) for k in ('left', 'top', 'width', 'height'))
            expected[(f, oid)] = list(map(int, a['rle'].split(','))) + [l, t, l+w-1, t+h-1]
        source = BASE / sid / 'sam2_raw_v1/masks_rle.jsonl.gz'
        original_nonempty = {}
        with gzip.open(source, 'rt', encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                if sum(r['rle'][:-4][1::2]):
                    original_nonempty[(r['frame'], r['id'])] = r['rle']
        assert expected == original_nonempty
        full_path = DEST / f'Aria_{sid}_SAM2_frame_masks.xml'
        ET.ElementTree(root).write(full_path, encoding='utf-8', xml_declaration=True)
        skip = []
        upload_path = full_path
        if sid == '164557':
            for track in old.findall('track'):
                first = track.find('mask')
                f, oid = int(first.attrib['frame']), int(track.attrib['id'])
                image = images[f]
                target = next(m for m in image.findall('mask')
                              if m.find('attribute').text == f'S{oid}')
                image.remove(target)
                skip.append({'frame': f, 'sam2_id': f'S{oid}'})
            upload_path = DEST / f'Aria_{sid}_SAM2_frame_masks_remainder.xml'
            ET.ElementTree(root).write(upload_path, encoding='utf-8', xml_declaration=True)
        parsed = ET.parse(upload_path)
        imported_count = len(parsed.findall('image/mask'))
        assert imported_count + len(skip) == len(expected)
        records.append({'recording': sid, 'full_xml': str(full_path), 'upload_xml': str(upload_path),
                        'nonempty_masks': len(expected), 'append_mask_count': imported_count,
                        'already_imported_prompt_masks': skip,
                        'all_raw_nonempty_rle_and_coordinates_exact': True,
                        'full_xml_sha256': hashlib.sha256(full_path.read_bytes()).hexdigest(),
                        'upload_xml_sha256': hashlib.sha256(upload_path.read_bytes()).hexdigest(),
                        'format': 'CVAT per-frame pixel mask shapes, immutable sam2_id',
                        'convert_masks_to_polygons': False, 'import_mode': 'append'})
    (DEST / 'frame_masks_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps([{'recording': r['recording'], 'append_masks': r['append_mask_count'],
                      'total_masks': r['nonempty_masks']} for r in records]))


if __name__ == '__main__':
    main()
