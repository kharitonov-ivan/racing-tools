#!/usr/bin/env python3
"""
Декодер бинарного формата .trackALFANO
Kartodrome Rustavi — Alfano GPS timing system

Структура файла (8192 байт фиксированный размер):
  0x00-0x04   Магик "*P* A"
  0x32        Версия прошивки (byte)
  0x33        Ревизия (byte)
  0x34-0x35   ID устройства (uint16 LE)
  0x36-0x3b   Служебные байты (сохраняются as-is)
  0x40-0x44   Название трассы (5 chars, пробел-паддинг, 0xFF = пусто)
  0x47-0x48   Полушарие ("NE" = North/East)
  0x50-0x53   Широта центра (int32 LE, микроградусы)
  0x54-0x57   Долгота центра (int32 LE, микроградусы)
  0x58-0x59   Параметр (uint16 LE, возможно длина трассы в м)
  0x5a-0x5f   Дополнительные параметры (uint16 LE x3)
  0x78-0x7a   Размер секции данных в байтах (uint24 LE = count × 11)
  0x7b+       Точки трассы по 11 байт, хвост 0xFF (паддинг до 8192)

Структура записи (11 байт):
  [0:4]  Широта    (int32 LE, микроградусы → / 1_000_000 = градусы)
  [4:8]  Долгота   (int32 LE, микроградусы → / 1_000_000 = градусы)
  [8:10] Неизвестно (uint16 LE, диапазон 512-1006, возможно высота)
  [10]   Скорость_raw (uint8, 0-199, 0 = стоп/вершина поворота, 199 = максимум)

Апексы (speed_raw < 10) — вершины поворотов:
  Внутренняя точка дуги виража с минимальной скоростью.
  Alfano использует их как контрольные точки секторов,
  для идентификации поворотов и валидации круга.
"""

import struct
import json
import csv
import sys
import math
from pathlib import Path


class AlfanoTrack:
    """
    Парсер и сериализатор бинарного формата .trackALFANO.

    Импорт:
        track = AlfanoTrack.from_file('path/to/file.trackALFANO')

    Экспорт:
        track.save_trackALFANO('output.trackALFANO')  # round-trip
        track.save_geojson('output.geojson')
        track.save_csv('output.csv')
    """

    FILE_SIZE   = 8192
    MAGIC       = b'*P* A'
    COORD_SCALE = 1_000_000
    RECORD_SIZE = 11
    DATA_OFFSET = 0x7B

    def __init__(self):
        self.header: dict = {}
        self.points: list[dict] = []
        self._raw_header: bytes = b''  # сырые байты 0x00-0x7A для round-trip

    # ─────────────────────────── IMPORT ────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> 'AlfanoTrack':
        """Читает .trackALFANO файл и возвращает объект AlfanoTrack."""
        data = Path(path).read_bytes()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AlfanoTrack':
        """Парсит бинарные данные .trackALFANO и возвращает объект AlfanoTrack."""
        track = cls()
        track.header = cls._parse_header(data)
        track.points = cls._parse_points(data, track.header['record_count'])
        track._raw_header = bytes(data[:cls.DATA_OFFSET])  # сохраняем все байты заголовка
        return track

    @staticmethod
    def _parse_header(data: bytes) -> dict:
        if not data.startswith(AlfanoTrack.MAGIC):
            raise ValueError(
                f"Неверный магик. Ожидалось {AlfanoTrack.MAGIC!r}, получено {data[:5]!r}"
            )

        hdr = {}
        hdr['magic']      = data[:5].decode('ascii')
        hdr['fw_version'] = data[0x32]
        hdr['fw_revision'] = data[0x33]
        hdr['device_id']  = struct.unpack_from('<H', data, 0x34)[0]
        hdr['flags']      = data[0x36:0x3C].hex()  # сохраняем as-is

        raw_name = data[0x40:0x45]
        hdr['track_name'] = raw_name.replace(b'\xff', b'').decode('ascii', errors='replace').strip()

        if data[0x47] != 0xFF:
            hdr['hemisphere'] = data[0x47:0x49].decode('ascii', errors='replace')

        hdr['center_lat']   = struct.unpack_from('<i', data, 0x50)[0] / AlfanoTrack.COORD_SCALE
        hdr['center_lon']   = struct.unpack_from('<i', data, 0x54)[0] / AlfanoTrack.COORD_SCALE

        hdr['param_0x58']   = struct.unpack_from('<H', data, 0x58)[0]
        hdr['param_0x5a']   = struct.unpack_from('<H', data, 0x5A)[0]
        hdr['param_0x5c']   = struct.unpack_from('<H', data, 0x5C)[0]
        hdr['param_0x5e']   = struct.unpack_from('<H', data, 0x5E)[0]

        data_section_bytes = int.from_bytes(data[0x78:0x7B], 'little')
        hdr['data_section_bytes'] = data_section_bytes
        hdr['record_count']       = data_section_bytes // AlfanoTrack.RECORD_SIZE

        return hdr

    @staticmethod
    def _parse_points(data: bytes, count: int) -> list[dict]:
        points = []
        for i in range(count):
            off = AlfanoTrack.DATA_OFFSET + i * AlfanoTrack.RECORD_SIZE
            if off + AlfanoTrack.RECORD_SIZE > len(data):
                break
            chunk = data[off:off + AlfanoTrack.RECORD_SIZE]
            if all(b == 0xFF for b in chunk):
                break

            lat_raw = struct.unpack_from('<i', chunk, 0)[0]
            lon_raw = struct.unpack_from('<i', chunk, 4)[0]
            unk_u16 = struct.unpack_from('<H', chunk, 8)[0]
            speed_raw = chunk[10]

            lat = lat_raw / AlfanoTrack.COORD_SCALE
            lon = lon_raw / AlfanoTrack.COORD_SCALE

            if not (40.0 < lat < 43.0 and 43.0 < lon < 47.0):
                print(f"  [!] Запись {i}: некорректные координаты lat={lat:.6f}, lon={lon:.6f} — остановка")
                break

            points.append({
                'index':     i,
                'lat':       lat,
                'lon':       lon,
                'unk_u16':   unk_u16,
                'speed_raw': speed_raw,
            })
        return points

    # ─────────────────────────── EXPORT ────────────────────────────

    def to_bytes(self) -> bytes:
        """
        Сериализует трассу обратно в бинарный .trackALFANO формат.
        Возвращает ровно FILE_SIZE (8192) байт с 0xFF-паддингом в хвосте.
        """
        buf = bytearray(b'\xff' * self.FILE_SIZE)
        hdr = self.header

        # Восстанавливаем сырой заголовок целиком (сохраняет все неизвестные байты)
        if self._raw_header:
            buf[:self.DATA_OFFSET] = self._raw_header

        # Перезаписываем известные изменяемые поля поверх сырого заголовка
        buf[0:5] = self.MAGIC
        buf[0x32] = hdr['fw_version']
        buf[0x33] = hdr['fw_revision']
        struct.pack_into('<H', buf, 0x34, hdr['device_id'])

        flags_bytes = bytes.fromhex(hdr['flags'])
        buf[0x36:0x36 + len(flags_bytes)] = flags_bytes

        name_bytes = hdr.get('track_name', '').encode('ascii', errors='replace')
        buf[0x40:0x45] = name_bytes[:5].ljust(5, b' ')

        hemi = hdr.get('hemisphere', '')
        if hemi:
            buf[0x47:0x49] = hemi.encode('ascii')[:2]

        struct.pack_into('<i', buf, 0x50, round(hdr['center_lat'] * self.COORD_SCALE))
        struct.pack_into('<i', buf, 0x54, round(hdr['center_lon'] * self.COORD_SCALE))

        struct.pack_into('<H', buf, 0x58, hdr['param_0x58'])
        struct.pack_into('<H', buf, 0x5A, hdr['param_0x5a'])
        struct.pack_into('<H', buf, 0x5C, hdr['param_0x5c'])
        struct.pack_into('<H', buf, 0x5E, hdr['param_0x5e'])

        # Размер секции данных (uint24 LE)
        data_len = len(self.points) * self.RECORD_SIZE
        buf[0x78:0x7B] = data_len.to_bytes(3, 'little')

        # Точки трассы
        for i, p in enumerate(self.points):
            off = self.DATA_OFFSET + i * self.RECORD_SIZE
            struct.pack_into('<i', buf, off,     round(p['lat'] * self.COORD_SCALE))
            struct.pack_into('<i', buf, off + 4, round(p['lon'] * self.COORD_SCALE))
            struct.pack_into('<H', buf, off + 8, p['unk_u16'])
            buf[off + 10] = p['speed_raw']

        return bytes(buf)

    def save_trackALFANO(self, path: str | Path):
        """Записывает бинарный .trackALFANO файл (round-trip)."""
        Path(path).write_bytes(self.to_bytes())

    def save_geojson(self, path: str | Path):
        """Экспортирует трассу в GeoJSON: линия, центр, апексы."""
        features = [
            {
                "type": "Feature",
                "properties": {
                    "name": self.header.get('track_name', ''),
                    "type": "track_line",
                    "points": len(self.points),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p['lon'], p['lat']] for p in self.points],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "center", "type": "center_point"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [self.header['center_lon'], self.header['center_lat']],
                },
            },
        ]

        for p in self.apexes:
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "apex",
                    "index": p['index'],
                    "speed_raw": p['speed_raw'],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [p['lon'], p['lat']],
                },
            })

        geojson = {"type": "FeatureCollection", "features": features}
        Path(path).write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding='utf-8')

    def save_csv(self, path: str | Path):
        """Экспортирует все точки трассы в CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'lat', 'lon', 'unk_u16', 'speed_raw'])
            writer.writeheader()
            writer.writerows(self.points)

    # ─────────────────────────── СВОЙСТВА ──────────────────────────

    @property
    def apexes(self) -> list[dict]:
        """
        Вершины поворотов — точки с минимальной скоростью (speed_raw < 10).
        В картинге «apex» — внутренняя точка дуги виража.
        Alfano использует их как контрольные точки секторов и для валидации круга.
        """
        return [p for p in self.points if p['speed_raw'] < 10]

    def stats(self) -> dict:
        """Возвращает статистику трассы: bounds, скорость, длина GPS-трека."""
        if not self.points:
            return {}
        lats   = [p['lat']       for p in self.points]
        lons   = [p['lon']       for p in self.points]
        speeds = [p['speed_raw'] for p in self.points]
        unk    = [p['unk_u16']   for p in self.points]

        def haversine(p1, p2):
            R = 6_371_000
            dlat = math.radians(p2['lat'] - p1['lat'])
            dlon = math.radians(p2['lon'] - p1['lon'])
            a = math.sin(dlat/2)**2 + math.cos(math.radians(p1['lat'])) * \
                math.cos(math.radians(p2['lat'])) * math.sin(dlon/2)**2
            return 2 * R * math.asin(math.sqrt(a))

        total_dist = sum(haversine(self.points[i], self.points[i+1])
                         for i in range(len(self.points) - 1))

        return {
            'lat_min':            min(lats),
            'lat_max':            max(lats),
            'lon_min':            min(lons),
            'lon_max':            max(lons),
            'speed_min':          min(speeds),
            'speed_max':          max(speeds),
            'speed_avg':          round(sum(speeds) / len(speeds), 1),
            'unk_min':            min(unk),
            'unk_max':            max(unk),
            'gps_track_length_m': round(total_dist, 1),
        }


# ─────────────────────────────── CLI ───────────────────────────────

def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path('/mnt/d/Downloads/CustomTracksAlfano/_P__A_ID10,008.trackALFANO')

    print(f"Файл: {input_path}")
    track = AlfanoTrack.from_file(input_path)
    print(f"Размер: {input_path.stat().st_size} байт\n")

    print("=== Заголовок ===")
    for k, v in track.header.items():
        print(f"  {k:25s}: {v}")

    print(f"\n=== Точки трассы ===")
    print(f"  Объявлено записей  : {track.header['record_count']}")
    print(f"  Разобрано точек    : {len(track.points)}")

    if not track.points:
        print("Нет данных!")
        return

    s = track.stats()
    print(f"\n=== Статистика ===")
    for k, v in s.items():
        print(f"  {k:25s}: {v}")

    print(f"\n=== Первые 5 точек ===")
    print(f"  {'idx':>4}  {'lat':>12}  {'lon':>12}  {'unk_u16':>8}  {'speed_raw':>10}")
    for p in track.points[:5]:
        print(f"  {p['index']:>4}  {p['lat']:>12.6f}  {p['lon']:>12.6f}  {p['unk_u16']:>8}  {p['speed_raw']:>10}")

    apexes = track.apexes
    print(f"\n=== Вершины поворотов (speed_raw < 10): {len(apexes)} точек ===")
    for p in apexes[:10]:
        print(f"  index={p['index']:3d}  lat={p['lat']:.6f}  lon={p['lon']:.6f}  speed={p['speed_raw']}")

    out_dir = input_path.parent
    stem    = input_path.stem

    geojson_path = out_dir / f"{stem}.geojson"
    track.save_geojson(geojson_path)
    print(f"\nGeoJSON сохранён: {geojson_path}")

    csv_path = out_dir / f"{stem}.csv"
    track.save_csv(csv_path)
    print(f"CSV сохранён    : {csv_path}")

    # Round-trip проверка
    rt_path = out_dir / f"{stem}_roundtrip.trackALFANO"
    track.save_trackALFANO(rt_path)
    original = input_path.read_bytes()
    roundtrip = rt_path.read_bytes()
    if original == roundtrip:
        print(f"Round-trip      : OK (побайтовое совпадение)")
    else:
        diff_offsets = [i for i in range(min(len(original), len(roundtrip)))
                        if original[i] != roundtrip[i]]
        print(f"Round-trip      : расхождения в {len(diff_offsets)} байтах: {diff_offsets[:10]}")
    rt_path.unlink()  # убираем временный файл


if __name__ == '__main__':
    main()
