#!/usr/bin/env python3
"""
Coordinator Server - Distributed Trading System
==============================================
REST API server that distributes work units to workers.

Features:
- Work unit distribution via REST API
- SQLite persistence
- Worker health monitoring
- Parallel backtesting support

Usage:
    python coordinator.py

API Endpoints:
    GET  /api/status           - System overview
    GET  /api/workers          - List workers
    GET  /api/get_work/<id>    - Get pending work
    POST /api/submit_result    - Submit result
    POST /api/create_work      - Create work unit
"""

import os
import sys
import json
import sqlite3
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - coordinator - %(levelname)s - %(message)s'
)
logger = logging.getLogger("coordinator")

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = "data/coordinator.db"
PORT = int(os.environ.get("COORDINATOR_PORT", 5001))
HOST = os.environ.get("COORDINATOR_HOST", "0.0.0.0")
WORKER_TIMEOUT_MINUTES = 5

# ============================================================================
# DATABASE
# ============================================================================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS work_units (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        status TEXT DEFAULT 'pending',
        work_type TEXT,
        params TEXT,
        replicas_needed INTEGER DEFAULT 1,
        replicas_assigned INTEGER DEFAULT 0,
        replicas_completed INTEGER DEFAULT 0,
        priority INTEGER DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        work_unit_id INTEGER,
        worker_id TEXT,
        created_at TEXT,
        pnl REAL,
        win_rate REAL,
        sharpe REAL,
        max_dd REAL,
        trades INTEGER,
        data TEXT,
        FOREIGN KEY (work_unit_id) REFERENCES work_units(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY,
        hostname TEXT,
        platform TEXT,
        status TEXT DEFAULT 'active',
        last_seen TEXT,
        work_units_completed INTEGER DEFAULT 0,
        total_execution_time REAL DEFAULT 0
    )''')
    
    conn.commit()
    return conn


# Global database connection
db_conn = init_db()
db_lock = threading.Lock()


# ============================================================================
# API ROUTES
# ============================================================================
@app.route('/api/status', methods=['GET'])
def status():
    """System status overview"""
    with db_lock:
        c = db_conn.cursor()
        
        # Work units stats
        c.execute("SELECT COUNT(*) FROM work_units WHERE status='pending'")
        pending = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM work_units WHERE status='in_progress'")
        in_progress = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM work_units WHERE status='completed'")
        completed = c.fetchone()[0]
        
        # Active workers (seen in last 5 minutes)
        cutoff = datetime.now().timestamp() - (WORKER_TIMEOUT_MINUTES * 60)
        c.execute("SELECT COUNT(*) FROM workers WHERE last_seen > ?", (cutoff,))
        active_workers = c.fetchone()[0]
        
        # Best result
        c.execute("SELECT MAX(pnl) FROM results")
        best_pnl = c.fetchone()[0] or 0
        
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "work_units": {
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed
        },
        "workers": {
            "active": active_workers
        },
        "best_pnl": best_pnl
    })


@app.route('/api/workers', methods=['GET'])
def workers():
    """List all workers"""
    with db_lock:
        c = db_conn.cursor()
        c.execute("SELECT * FROM workers ORDER BY last_seen DESC")
        columns = [desc[0] for desc in c.description]
        workers = []
        
        for row in c.fetchall():
            w = dict(zip(columns, row))
            # Convert last_seen to readable format
            try:
                ts = datetime.fromisoformat(w['last_seen'])
                w['last_seen_formatted'] = ts.strftime('%Y-%m-%d %H:%M:%S')
                w['minutes_ago'] = (datetime.now() - ts).total_seconds() / 60
            except:
                w['last_seen_formatted'] = w['last_seed']
                w['minutes_ago'] = 999
            
            w['alive'] = w['minutes_ago'] < WORKER_TIMEOUT_MINUTES
            workers.append(w)
    
    return jsonify({"workers": workers})


@app.route('/api/register_worker', methods=['POST'])
def register_worker():
    """Register or update a worker"""
    data = request.json
    
    worker_id = request.json.get('worker_id')
    hostname = data.get('hostname', 'unknown')
    platform = data.get('platform', 'unknown')
    
    with db_lock:
        c = db_conn.cursor()
        c.execute('''INSERT OR REPLACE INTO workers 
            (id, hostname, platform, status, last_seen) 
            VALUES (?, ?, ?, 'active', ?)''',
            (worker_id, hostname, platform, datetime.now().isoformat())
        )
        db_conn.commit()
    
    logger.info(f"Worker registered: {worker_id} ({hostname})")
    return jsonify({"status": "registered", "worker_id": worker_id})


@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """Worker heartbeat"""
    worker_id = request.json.get('worker_id')
    
    with db_lock:
        c = db_conn.cursor()
        c.execute("UPDATE workers SET last_seen = ? WHERE id = ?", 
                  (datetime.now().isoformat(), worker_id))
        db_conn.commit()
    
    return jsonify({"status": "ok"})


@app.route('/api/get_work/<worker_id>', methods=['GET'])
def get_work(worker_id):
    """
    Worker requests work.
    Returns pending work unit or None if all assigned.
    """
    with db_lock:
        c = db_conn.cursor()
        
        # Update worker last_seen
        c.execute("UPDATE workers SET last_seen = ? WHERE id = ?", 
                  (datetime.now().isoformat(), worker_id))
        
        # Find pending work (replicas_assigned < replicas_needed)
        c.execute('''SELECT id, work_type, params, replicas_needed, replicas_assigned 
            FROM work_units 
            WHERE status IN ('pending', 'in_progress') 
            AND replicas_assigned < replicas_needed
            ORDER BY priority DESC, created_at ASC
            LIMIT 1''')
        
        row = c.fetchone()
        
        if not row:
            return jsonify({"work": None, "message": "No work available"})
        
        work_id, work_type, params, replicas_needed, replicas_assigned = row
        
        # Increment assigned count immediately (prevent race)
        c.execute("UPDATE work_units SET replicas_assigned = ?, status = 'in_progress' WHERE id = ?",
                  (replicas_needed, work_id))
        db_conn.commit()
        
        # Log work assignment
        c.execute("INSERT INTO work_assignments (work_unit_id, worker_id, assigned_at) VALUES (?, ?, ?)",
                  (work_id, worker_id, datetime.now().isoformat()))
        db_conn.commit()
        
        return jsonify({
            "work": {
                "id": work_id,
                "work_type": work_type,
                "params": json.loads(params),
                "replicas_needed": replicas_needed,
                "replicas_assigned": replicas_needed
            }
        })


@app.route('/api/submit_result', methods=['POST'])
def submit_result():
    """Worker submits backtest result"""
    data = request.json
    
    work_unit_id = data.get('work_unit_id')
    worker_id = request.json.get('worker_id')
    pnl = data.get('pnl', 0)
    win_rate = data.get('win_rate', 0)
    sharpe = data.get('sharpe', 0)
    max_dd = data.get('max_dd', 0)
    trades = data.get('trades', 0)
    result_data = data.get('data', '{}')
    
    with db_lock:
        c = db_conn.cursor()
        
        # Save result
        c.execute('''INSERT INTO results 
            (work_unit_id, worker_id, created_at, pnl, win_rate, sharpe, max_dd, trades, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (work_unit_id, worker_id, datetime.now().isoformat(), 
             pnl, win_rate, sharpe, max_dd, trades, json.dumps(result_data)))
        
        # Update completion count
        c.execute("UPDATE work_units SET replicas_completed = replicas_completed + 1 WHERE id = ?", (work_unit_id,))
        
        # Check if complete
        c.execute("SELECT replicas_needed, replicas_completed FROM work_units WHERE id = ?", (work_unit_id,))
        needed, completed = c.fetchone()
        
        if completed >= needed:
            c.execute("UPDATE work_units SET status = 'completed' WHERE id = ?", (work_unit_id,))
            logger.info(f"Work unit {work_unit_id} COMPLETED")
        
        # Update worker stats
        c.execute("""UPDATE workers SET 
            work_units_completed = work_units_completed + 1 
            WHERE id = ?""", (worker_id,))
        
        db_conn.commit()
    
    logger.info(f"Result from {worker_id}: PnL={pnl:.4f}")
    return jsonify({"status": "saved", "work_unit_id": work_unit_id})


@app.route('/api/create_work', methods=['POST'])
def create_work():
    """Create a new work unit"""
    data = request.json
    
    work_type = data.get('work_type', 'backtest')
    params = data.get('params', {})
    replicas = data.get('replicas', 1)
    priority = data.get('priority', 0)
    
    with db_lock:
        c = db_conn.cursor()
        c.execute('''INSERT INTO work_units 
            (created_at, work_type, params, replicas_needed, replicas_assigned, replicas_completed, priority)
            VALUES (?, ?, ?, ?, 0, 0, ?)''',
            (datetime.now().isoformat(), work_type, json.dumps(params), replicas, priority))
        
        work_id = c.lastrowid
        db_conn.commit()
    
    logger.info(f"Created work unit {work_id}: {work_type}")
    return jsonify({"status": "created", "id": work_id})


@app.route('/api/results', methods=['GET'])
def results():
    """Get all completed results"""
    with db_lock:
        c = db_conn.cursor()
        c.execute('''SELECT r.*, w.work_type 
            FROM results r 
            JOIN work_units w ON r.work_unit_id = w.id 
            ORDER BY r.pnl DESC 
            LIMIT 50''')
        
        columns = [desc[0] for desc in c.description]
        results = []
        for row in c.fetchall():
            r = dict(zip(columns, row))
            r['params'] = json.loads(r.get('data', '{}'))
            results.append(r)
    
    return jsonify({"results": results})


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Reset stuck work units"""
    with db_lock:
        c = db_conn.cursor()
        c.execute('''UPDATE work_units SET status = 'pending', 
            replicas_assigned = replicas_completed 
            WHERE status = 'in_progress' 
            AND replicas_assigned <= replicas_completed''')
        db_conn.commit()
    
    return jsonify({"status": "cleaned"})


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ============================================================================
# MAIN
# ============================================================================
def run():
    """Run the coordinator server"""
    logger.info(f"🚀 Starting Coordinator on {HOST}:{PORT}")
    logger.info(f"📂 Database: {DB_PATH}")
    
    # Create data directory if needed
    os.makedirs("data", exist_ok=True)
    
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    run()
