#!/usr/bin/env python3
"""
Backup Manager para Solana Trading Bot
Previene pérdida de datos críticos como el autoaprendizaje
"""
import os
import json
import shutil
import sqlite3
from datetime import datetime, timedelta
import gzip
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupManager:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.backup_dir = os.path.join(self.data_dir, 'backups')
        self.critical_files = [
            'auto_learner.db',
            'trade_history.json',
            'portfolio.json',
            'auto_learner_state.json',
            'parameter_optimization.json'
        ]
        
        # Crear directorio de backups si no existe
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def backup_all(self):
        """Backup completo de todos los archivos críticos"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = os.path.join(self.backup_dir, f'full_backup_{timestamp}')
        os.makedirs(backup_subdir, exist_ok=True)
        
        backed_up = []
        for file in self.critical_files:
            src = os.path.join(self.data_dir, file)
            if os.path.exists(src):
                dst = os.path.join(backup_subdir, file)
                shutil.copy2(src, dst)
                backed_up.append(file)
                logger.info(f"Backed up: {file}")
        
        # Comprimir backups antiguos (> 7 días)
        self._compress_old_backups()
        
        logger.info(f"Full backup completed: {len(backed_up)} files backed up to {backup_subdir}")
        return backup_subdir
        
    def backup_incremental(self, trade_count=None):
        """Backup incremental cuando se alcanzan hitos (cada 100 trades)"""
        if trade_count and trade_count % 100 == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Backup solo archivos de trades y aprendizaje
            for file in ['trade_history.json', 'auto_learner.db']:
                src = os.path.join(self.data_dir, file)
                if os.path.exists(src):
                    dst = os.path.join(
                        self.backup_dir, 
                        f'{file.split(".")[0]}_snapshot_{trade_count}trades_{timestamp}.{file.split(".")[-1]}'
                    )
                    shutil.copy2(src, dst)
                    logger.info(f"Incremental backup: {file} at {trade_count} trades")
                    
    def backup_pre_operation(self, operation_name):
        """Backup antes de operaciones críticas"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = os.path.join(self.backup_dir, f'pre_{operation_name}_{timestamp}')
        os.makedirs(backup_subdir, exist_ok=True)
        
        # Backup y crear log de la operación
        for file in self.critical_files:
            src = os.path.join(self.data_dir, file)
            if os.path.exists(src):
                dst = os.path.join(backup_subdir, file)
                shutil.copy2(src, dst)
                
        # Log de la operación
        log_data = {
            'operation': operation_name,
            'timestamp': timestamp,
            'reason': 'Pre-operation backup',
            'files_backed_up': [f for f in self.critical_files if os.path.exists(os.path.join(self.data_dir, f))]
        }
        
        with open(os.path.join(backup_subdir, 'operation_log.json'), 'w') as f:
            json.dump(log_data, f, indent=2)
            
        logger.warning(f"Pre-operation backup for '{operation_name}' saved to {backup_subdir}")
        return backup_subdir
        
    def restore_from_date(self, date_str):
        """Restaurar desde una fecha específica (formato: YYYYMMDD)"""
        # Buscar el backup más cercano a la fecha
        backups = sorted([d for d in os.listdir(self.backup_dir) if d.startswith('full_backup_')])
        
        for backup in reversed(backups):
            if date_str in backup:
                backup_path = os.path.join(self.backup_dir, backup)
                return self._restore_from_path(backup_path)
                
        logger.error(f"No backup found for date: {date_str}")
        return False
        
    def restore_latest(self):
        """Restaurar desde el último backup disponible"""
        backups = sorted([d for d in os.listdir(self.backup_dir) if d.startswith('full_backup_')])
        
        if backups:
            latest = os.path.join(self.backup_dir, backups[-1])
            return self._restore_from_path(latest)
        else:
            logger.error("No backups found!")
            return False
            
    def _restore_from_path(self, backup_path):
        """Restaurar desde un path específico"""
        if not os.path.exists(backup_path):
            logger.error(f"Backup path not found: {backup_path}")
            return False
            
        # Crear backup del estado actual antes de restaurar
        self.backup_pre_operation('restore')
        
        restored = []
        for file in self.critical_files:
            src = os.path.join(backup_path, file)
            dst = os.path.join(self.data_dir, file)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                restored.append(file)
                logger.info(f"Restored: {file} from {backup_path}")
                
        logger.info(f"Restoration completed: {len(restored)} files restored")
        return True
        
    def verify_backups(self):
        """Verificar integridad de los backups"""
        issues = []
        
        # Verificar cada backup
        for backup_dir_name in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_dir_name)
            
            if os.path.isdir(backup_path):
                # Verificar archivos JSON
                for file in os.listdir(backup_path):
                    if file.endswith('.json'):
                        filepath = os.path.join(backup_path, file)
                        try:
                            with open(filepath, 'r') as f:
                                json.load(f)
                        except Exception as e:
                            issues.append(f"{filepath}: {str(e)}")
                            
                # Verificar base de datos
                if 'auto_learner.db' in os.listdir(backup_path):
                    db_path = os.path.join(backup_path, 'auto_learner.db')
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.execute("SELECT COUNT(*) FROM trade_results")
                        conn.close()
                    except Exception as e:
                        issues.append(f"{db_path}: {str(e)}")
                        
        if issues:
            logger.warning(f"Backup verification found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("All backups verified successfully")
            
        return len(issues) == 0
        
    def _compress_old_backups(self):
        """Comprimir backups más antiguos de 7 días"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for backup_name in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            if os.path.isdir(backup_path) and backup_name.startswith('full_backup_'):
                # Extraer fecha del nombre
                try:
                    date_str = backup_name.split('_')[2]
                    backup_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if backup_date < cutoff_date and not backup_name.endswith('.tar.gz'):
                        # Comprimir
                        archive_path = f"{backup_path}.tar.gz"
                        shutil.make_archive(backup_path, 'gztar', backup_path)
                        shutil.rmtree(backup_path)
                        logger.info(f"Compressed old backup: {backup_name}")
                except Exception as e:
                    logger.warning(f"Could not process backup {backup_name}: {e}")
                    
    def cleanup_old(self, days=30):
        """Eliminar backups más antiguos de X días"""
        cutoff_date = datetime.now() - timedelta(days=days)
        removed = 0
        
        for backup_name in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            try:
                # Extraer fecha del nombre
                if 'full_backup_' in backup_name:
                    date_str = backup_name.split('_')[2].split('.')[0]
                    backup_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if backup_date < cutoff_date:
                        if os.path.isdir(backup_path):
                            shutil.rmtree(backup_path)
                        else:
                            os.remove(backup_path)
                        removed += 1
                        logger.info(f"Removed old backup: {backup_name}")
            except Exception as e:
                logger.warning(f"Could not remove {backup_name}: {e}")
                
        logger.info(f"Cleanup completed: {removed} old backups removed")
        return removed


# Script para uso directo
if __name__ == "__main__":
    import sys
    
    manager = BackupManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "backup":
            manager.backup_all()
        elif command == "restore":
            if len(sys.argv) > 2:
                manager.restore_from_date(sys.argv[2])
            else:
                manager.restore_latest()
        elif command == "verify":
            manager.verify_backups()
        elif command == "cleanup":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            manager.cleanup_old(days)
        else:
            print("Usage: python backup_manager.py [backup|restore|verify|cleanup] [options]")
    else:
        # Por defecto, hacer backup completo
        manager.backup_all()