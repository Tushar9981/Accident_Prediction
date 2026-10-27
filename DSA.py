from flask import Flask, render_template, jsonify, request
import collections
import heapq
import threading
import time
import random

app = Flask(__name__)

# Global thread lock for state safety across API and background threads
state_lock = threading.Lock()

class Vehicle:
    """Class representing a single vehicle in the traffic system."""
    def __init__(self, vehicle_id, lane, vehicle_type, arrival_time):
        self.id = vehicle_id
        self.lane = lane.upper()          # 'N', 'S', 'E', 'W'
        self.type = vehicle_type.upper()  # 'NORMAL' or 'EMERGENCY'
        self.arrival_time = arrival_time
        # Priority value for Min-Heap comparison: 0 for Emergency, 1 for Normal
        self.priority = 0 if self.type == 'EMERGENCY' else 1

    def __lt__(self, other):
        """Comparator for heapq priority queue (Emergency vehicles pop first, then by arrival time)."""
        if self.priority == other.priority:
            return self.arrival_time < other.arrival_time
        return self.priority < other.priority

    def to_dict(self):
        return {
            'id': self.id,
            'lane': self.lane,
            'type': self.type,
            'arrival_time': round(self.arrival_time, 1)
        }

class TrafficController:
    """Core Traffic Controller managing Lane Queues, Priority Queue, and Signal Timers."""
    def __init__(self):
        self.lanes_order = ['N', 'E', 'S', 'W']
        self.lane_index = 0
        self.active_lane = 'N'
        self.signal_state = 'GREEN'  # 'GREEN' or 'YELLOW'
        
        # Simulation Modes
        self.auto_spawn = False
        self.spawn_timer = 0.0
        
        # Data Structures: 4 FIFO Queues & 1 Priority Queue (Heap)
        self.queues = {
            'N': collections.deque(),
            'E': collections.deque(),
            'S': collections.deque(),
            'W': collections.deque()
        }
        self.emergency_pq = []  # Heap storing tuples: (priority, arrival_time, vehicle_id, vehicle_object)
        
        # Timing Parameters (Seconds)
        self.t_base = 5.0
        self.t_clear_per_car = 2.0
        self.t_min = 10.0
        self.t_max = 60.0
        self.yellow_duration = 2.0
        
        # Phase Timers
        self.phase_duration = self.calculate_green_time('N')
        self.remaining_time = self.phase_duration
        
        # Metrics & System Counters
        self.vehicle_counter = 1
        self.total_cleared = 0
        self.total_wait_time = 0.0
        self.simulation_clock = 0.0
        self.last_clear_time = 0.0
        
        # Real-time Event Logger
        self.logs = []
        self.add_log("System initialized. Active Phase: Lane N GREEN. Mode: Manual Traffic Entry.")

    def add_log(self, message):
        timestamp = f"[{self.simulation_clock:.1f}s]"
        self.logs.append(f"{timestamp} {message}")
        if len(self.logs) > 40:
            self.logs.pop(0)

    def calculate_green_time(self, lane):
        """Dynamic timing formula based on lane queue density."""
        q_len = len(self.queues[lane])
        calc_time = self.t_base + (q_len * self.t_clear_per_car)
        return max(self.t_min, min(self.t_max, calc_time))

    def add_vehicle(self, lane, is_emergency=False):
        """Enqueues a new vehicle into FIFO lane queue and optional Priority Queue."""
        lane = lane.upper()
        if lane not in self.queues:
            return None
        
        if isinstance(is_emergency, str):
            is_emergency = is_emergency.lower() in ('true', '1', 'yes')
        
        v_type = 'EMERGENCY' if is_emergency else 'NORMAL'
        prefix = 'AMB' if is_emergency else lane
        v_id = f"{prefix}-{self.vehicle_counter:03d}"
        self.vehicle_counter += 1
        
        v = Vehicle(v_id, lane, v_type, self.simulation_clock)
        
        if is_emergency:
            self.queues[lane].appendleft(v)
            heapq.heappush(self.emergency_pq, (0, self.simulation_clock, v_id, v))
            self.add_log(f"ALERT: Emergency Vehicle {v_id} queued in Lane {lane}! Heap Priority active.")
        else:
            self.queues[lane].append(v)
            self.add_log(f"Vehicle {v_id} arrived in Lane {lane} (Queue length: {len(self.queues[lane])}).")
        
        return v

    def _remove_from_emergency_heap(self, vehicle_id):
        """Helper to purge a vehicle from emergency priority queue by ID and re-heapify."""
        initial_len = len(self.emergency_pq)
        self.emergency_pq = [item for item in self.emergency_pq if item[3].id != vehicle_id]
        if len(self.emergency_pq) < initial_len:
            heapq.heapify(self.emergency_pq)
            return True
        return False

    def tick(self, dt=0.2):
        """Main simulation step called periodically."""
        self.simulation_clock += dt
        self.remaining_time -= dt

        # Handle Auto-Spawning if enabled (spawns a random vehicle every ~2 seconds)
        if self.auto_spawn:
            self.spawn_timer += dt
            if self.spawn_timer >= 2.0:
                self.spawn_timer = 0.0
                random_lane = random.choice(self.lanes_order)
                # 10% chance to spawn an emergency vehicle
                is_emg = random.random() < 0.1
                self.add_vehicle(random_lane, is_emg)

        # Emergency Preemption Check (During GREEN phase)
        if self.signal_state == 'GREEN' and self.emergency_pq:
            top_emg = self.emergency_pq[0][3]
            if top_emg.lane != self.active_lane and self.remaining_time > self.yellow_duration:
                self.add_log(f"PREEMPTION TRIGGERED! Interrupting Lane {self.active_lane} for Emergency in Lane {top_emg.lane}.")
                self.signal_state = 'YELLOW'
                self.remaining_time = self.yellow_duration

        # Dequeue vehicle from active GREEN lane at clearance rate
        if self.signal_state == 'GREEN' and self.queues[self.active_lane]:
            if (self.simulation_clock - self.last_clear_time) >= 1.5:
                self.last_clear_time = self.simulation_clock
                cleared_v = self.queues[self.active_lane].popleft()
                wait = self.simulation_clock - cleared_v.arrival_time
                self.total_cleared += 1
                self.total_wait_time += wait
                
                was_emg = self._remove_from_emergency_heap(cleared_v.id)
                if cleared_v.type == 'EMERGENCY' or was_emg:
                    self.add_log(f"PRIORITY CLEARED: Emergency {cleared_v.id} cleared Lane {self.active_lane}. Priority Heap updated ({len(self.emergency_pq)} active).")
                else:
                    self.add_log(f"DEQUEUE (FIFO): Vehicle {cleared_v.id} cleared Lane {self.active_lane}. Wait: {wait:.1f}s.")

        # Phase Transition Evaluation
        if self.remaining_time <= 0:
            if self.signal_state == 'GREEN':
                self.signal_state = 'YELLOW'
                self.remaining_time = self.yellow_duration
                self.add_log(f"Signal Lane {self.active_lane} -> YELLOW ({self.yellow_duration:.1f}s).")
            elif self.signal_state == 'YELLOW':
                if self.emergency_pq:
                    _, _, _, emg_v = self.emergency_pq[0]
                    next_lane = emg_v.lane
                    self.add_log(f"PRIORITY OVERRIDE: Fast-tracking Lane {next_lane} for Emergency Vehicle!")
                else:
                    self.lane_index = (self.lane_index + 1) % 4
                    next_lane = self.lanes_order[self.lane_index]

                self.active_lane = next_lane
                self.lane_index = self.lanes_order.index(next_lane)
                self.signal_state = 'GREEN'
                self.phase_duration = self.calculate_green_time(next_lane)
                self.remaining_time = self.phase_duration
                self.add_log(f"Phase Change: Lane {next_lane} -> GREEN for {self.phase_duration:.1f}s (Queue: {len(self.queues[next_lane])}).")

    def reset(self):
        """Resets simulation queues, metrics, and signal timers."""
        self.lane_index = 0
        self.active_lane = 'N'
        self.signal_state = 'GREEN'
        self.auto_spawn = False
        self.spawn_timer = 0.0
        self.queues = {'N': collections.deque(), 'E': collections.deque(), 'S': collections.deque(), 'W': collections.deque()}
        self.emergency_pq = []
        self.vehicle_counter = 1
        self.total_cleared = 0
        self.total_wait_time = 0.0
        self.simulation_clock = 0.0
        self.phase_duration = self.calculate_green_time('N')
        self.remaining_time = self.phase_duration
        self.logs.clear()
        self.add_log("System state reset successfully. Waiting for manual vehicle entry.")

    def get_state(self):
        """Exports current state dictionary for Flask JSON API."""
        signals = {}
        for l in self.lanes_order:
            if l == self.active_lane:
                signals[l] = self.signal_state
            else:
                signals[l] = 'RED'

        avg_wait = (self.total_wait_time / self.total_cleared) if self.total_cleared > 0 else 0.0
        emg_list = [item[3].to_dict() for item in sorted(self.emergency_pq)]

        return {
            'active_lane': self.active_lane,
            'signal_state': self.signal_state,
            'signals': signals,
            'remaining_time': max(0.0, round(self.remaining_time, 1)),
            'phase_duration': round(self.phase_duration, 1),
            'queues': {l: [v.to_dict() for v in self.queues[l]] for l in self.lanes_order},
            'emergency_queue': emg_list,
            'metrics': {
                'total_cleared': self.total_cleared,
                'total_wait_time': round(self.total_wait_time, 1),
                'avg_wait_time': round(avg_wait, 1),
                'simulation_clock': round(self.simulation_clock, 1)
            },
            'auto_spawn': self.auto_spawn,
            'logs': self.logs[-20:]
        }

# Global Controller Instance
controller = TrafficController()

def background_simulation():
    """Background thread running continuous simulation ticks."""
    while True:
        with state_lock:
            controller.tick(dt=0.2)
        time.sleep(0.2)

# Start background simulation thread
sim_thread = threading.Thread(target=background_simulation, daemon=True)
sim_thread.start()

# --- Flask Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    with state_lock:
        return jsonify(controller.get_state())

@app.route('/api/add_vehicle', methods=['POST'])
def add_vehicle():
    data = request.json or {}
    lane = data.get('lane', 'N')
    is_emergency = data.get('is_emergency', False)
    with state_lock:
        v = controller.add_vehicle(lane, is_emergency)
        return jsonify({'success': True, 'vehicle': v.to_dict() if v else None})

@app.route('/api/auto_spawn', methods=['POST'])
def toggle_auto_spawn():
    with state_lock:
        controller.auto_spawn = not controller.auto_spawn
        status = "enabled" if controller.auto_spawn else "disabled"
        controller.add_log(f"Auto-Spawn toggled: {status}.")
        return jsonify({'success': True, 'auto_spawn': controller.auto_spawn})

@app.route('/api/reset', methods=['POST'])
def reset_system():
    with state_lock:
        controller.reset()
        return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting Traffic Signal Control Simulator on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
