#!/usr/bin/env python3
"""Racing Monitor - Listen to racing timing system and display timetable."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

import re
import requests
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path


@dataclass
class Driver:
    """Driver information"""
    kart_number: str
    number: str
    position: int
    name: str
    team: str = ""
    _class: str = ""
    group: int = 0


@dataclass
class LapTime:
    """Lap time information"""
    kart_number: str
    laps: int
    time: str  # Format: "HH:MM:SS.mmm" or "00:00:00"
    position: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """Race session information"""
    name: str
    type: str  # 'H' (Heat), 'G' (Qualifying), 'SP' (Sprint Practice), 'SR' (Sprint Race)
    laps: Dict[str, LapTime] = field(default_factory=dict)

    def get_leaderboard(self) -> List[LapTime]:
        """Get sorted leaderboard"""
        valid_laps = [lap for lap in self.laps.values() if lap.laps > 0]
        return sorted(valid_laps, key=lambda x: (-x.laps, self._parse_time(x.time)))

    @staticmethod
    def _parse_time(time_str: str) -> timedelta:
        """Parse time string to timedelta"""
        if time_str == "00:00:00":
            return timedelta(days=1)  # Max time for no time
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return timedelta(hours=int(h), minutes=int(m), seconds=float(s))
        except:
            pass
        return timedelta(days=1)


class RacingDataParser:
    """Parser for racing timing system data stream"""

    def __init__(self):
        self.drivers: Dict[str, Driver] = {}
        self.sessions: Dict[str, Session] = {
            'H': Session('Heat', 'H'),
            'G': Session('Qualifying', 'G'),
            'SP': Session('Sprint Practice', 'SP'),
            'SR': Session('Sprint Race', 'SR'),
        }
        self.classes: Dict[int, str] = {}
        self.track_name = "Unknown Track"
        self.track_length = 0.0
        self.current_run = 0
        self.session_types = {
            '$H': 'H',
            '$G': 'G',
            '$SP': 'SP',
            '$SR': 'SR'
        }

    def parse_line(self, line: str) -> Optional[str]:
        """Parse a single line from the data stream"""
        if not line or not line.startswith('$'):
            return None

        parts = line.split(',')
        msg_type = parts[0]

        # Parse driver/competitor info ($A)
        if msg_type == '$A':
            self._parse_driver(parts)
            return f"Driver: {parts[3]} (Kart {parts[1]})"

        # Parse competitor info ($COMP)
        elif msg_type == '$COMP':
            self._parse_competitor(parts)
            return None

        # Parse timing data
        elif msg_type in self.session_types:
            session_type = self.session_types[msg_type]
            return self._parse_timing(parts, session_type)

        # Parse class info ($C)
        elif msg_type == '$C':
            self._parse_class(parts)
            return None

        # Parse event/track info ($E)
        elif msg_type == '$E':
            return self._parse_event_info(parts)

        # Parse new run ($B)
        elif msg_type == '$B':
            self.current_run += 1
            return f"=== New Run #{self.current_run} ==="

        # Parse finish ($F)
        elif msg_type == '$F':
            return f"=== Session Finished at {parts[2]} ==="

        return None

    def _parse_driver(self, parts: list):
        """Parse driver information: $A,kart_num,number,pos,name,team,class,group"""
        if len(parts) >= 8:
            self.drivers[parts[1]] = Driver(
                kart_number=parts[1],
                number=parts[2],
                position=int(parts[3]),
                name=parts[4],
                team=parts[5],
                _class=parts[6],
                group=int(parts[7]) if parts[7].isdigit() else 0
            )

    def _parse_competitor(self, parts: list):
        """Parse competitor info (same format as $A but for confirmation)"""
        if len(parts) >= 4:
            kart_num = parts[1]
            if kart_num not in self.drivers:
                self.drivers[kart_num] = Driver(
                    kart_number=kart_num,
                    number=parts[2],
                    position=int(parts[3]),
                    name=parts[4],
                    _class=parts[6] if len(parts) > 6 else "",
                    group=int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else 0
                )

    def _parse_timing(self, parts: list, session_type: str) -> str:
        """Parse timing data: $TYPE,pos,kart_num,laps,time"""
        if len(parts) >= 5:
            pos = int(parts[1])
            kart_num = parts[2].strip('"')
            laps = int(parts[3]) if parts[3] else 0
            time_str = parts[4].strip('"')

            lap_time = LapTime(
                kart_number=kart_num,
                position=pos,
                laps=laps,
                time=time_str
            )

            self.sessions[session_type].laps[kart_num] = lap_time

            # Get driver name if available
            driver_name = self.drivers.get(kart_num, Driver(kart_num, "", 0, "")).name
            return f"{self.sessions[session_type].name}: {pos}. Kart {kart_num} ({driver_name}) - {laps} laps - {time_str}"

        return ""

    def _parse_class(self, parts: list):
        """Parse class info: $C,id,name"""
        if len(parts) >= 3:
            class_id = int(parts[1])
            class_name = parts[2].strip('"')
            if class_name:
                self.classes[class_id] = class_name

    def _parse_event_info(self, parts: list):
        """Parse event/track info: $E,key,value"""
        if len(parts) >= 3:
            key = parts[1].strip('"')
            value = parts[2].strip('"')

            if key == 'TRACKNAME':
                self.track_name = value
            elif key == 'TRACKLENGTH':
                try:
                    self.track_length = float(value)
                except:
                    pass

            return f"Track: {self.track_name} ({self.track_length} km)"

        return None

    def get_session_leaderboard(self, session_type: str, limit: int = 20) -> List[tuple]:
        """Get formatted leaderboard for a session"""
        if session_type not in self.sessions:
            return []

        session = self.sessions[session_type]
        leaderboard = session.get_leaderboard()

        results = []
        for i, lap in enumerate(leaderboard[:limit], 1):
            driver = self.drivers.get(lap.kart_number, Driver(lap.kart_number, "", 0, ""))
            results.append((
                i,
                lap.kart_number,
                driver.name,
                lap.laps,
                lap.time
            ))

        return results


def print_leaderboard(parser: RacingDataParser, session_type: str, title: str):
    """Print a formatted leaderboard"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Pos':<6} {'Kart':<6} {'Driver':<30} {'Laps':<6} {'Best Time':<15}")
    print(f"{'-'*80}")

    leaderboard = parser.get_session_leaderboard(session_type)

    for pos, kart, name, laps, time in leaderboard:
        print(f"{pos:<6} {kart:<6} {name:<30} {laps:<6} {time:<15}")

    if not leaderboard:
        print("No data available yet...")

    print(f"{'='*80}\n")


def main():
    """Main application"""
    print("Racing Monitor - Connecting to http://10.10.31.20:50000/")
    print("Press Ctrl+C to stop\n")

    parser = RacingDataParser()

    try:
        # Connect to the stream
        response = requests.get('http://10.10.31.20:50000/', stream=True, timeout=None)
        response.raise_for_status()

        print("✓ Connected! Monitoring race data...\n")

        # Read and parse lines
        for line in response.iter_lines(decode_unicode=True):
            if line:
                line = line.strip()
                if line:
                    update = parser.parse_line(line)
                    if update:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {update}")

                    # Show leaderboards periodically
                    if line.startswith('$B') or line.startswith('$F'):
                        print_leaderboard(parser, 'H', 'HEAT - CURRENT STANDINGS')
                        print_leaderboard(parser, 'G', 'QUALIFYING - CURRENT STANDINGS')
                        print_leaderboard(parser, 'SP', 'SPRINT PRACTICE - CURRENT STANDINGS')
                        print_leaderboard(parser, 'SR', 'SPRINT RACE - CURRENT STANDINGS')

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        print("\n=== FINAL RESULTS ===")
        print_leaderboard(parser, 'H', 'HEAT - FINAL RESULTS')
        print_leaderboard(parser, 'G', 'QUALIFYING - FINAL RESULTS')
        print_leaderboard(parser, 'SP', 'SPRINT PRACTICE - FINAL RESULTS')
        print_leaderboard(parser, 'SR', 'SPRINT RACE - FINAL RESULTS')

        print(f"\nTrack: {parser.track_name}")
        print(f"Total Drivers: {len(parser.drivers)}")
        print(f"Classes: {', '.join(parser.classes.values())}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
