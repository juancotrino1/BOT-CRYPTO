"""
Sistema de Trading Algorítmico Mejorado
Versión: 2.0 - Producción Ready

MEJORAS PRINCIPALES:
- Validación estadística robusta
- Costos de transacción realistas
- Logging estructurado
- Gestión de errores mejorada
- Prevención de overfitting
- Backtesting realista
- Monitoreo de modelo
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import requests
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import json
import time
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
import sqlite3
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================

def setup_logging():
    """Configura logging estructurado"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'trading_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================
# CONFIGURACIÓN CON VALIDACIÓN
# ============================================

@dataclass
class TradingConfig:
    """Configuración validada para el sistema"""
    
    # Zona horaria
    TIMEZONE: pytz.timezone = pytz.timezone('America/Bogota')
    
    # Yahoo Finance: datos 1h limitados a 730 días
    INTERVALO: str = "1h"
    
    # Periodos (ajustados a límites)
    DIAS_ENTRENAMIENTO: int = 300  # ~10 meses
    DIAS_VALIDACION: int = 60      # 2 meses
    DIAS_BACKTEST: int = 30        # 1 mes
    
    # Activos (reducido para testing)
    ACTIVOS: List[str] = None
    
    def __post_init__(self):
        if self.ACTIVOS is None:
            self.ACTIVOS = [
                "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", 
                "LINK-USD", "AAVE-USD", "NEAR-USD"
            ]
    
    # Features técnicas
    VENTANA_VOLATILIDAD: int = 24
    RSI_PERIODO: int = 14
    ATR_PERIODO: int = 14
    
    # Horizontes (solo los más estables)
    HORIZONTES: List[int] = None
    
    def __post_init__(self):
        if self.HORIZONTES is None:
            self.HORIZONTES = [4, 8]  # 4h y 8h son más estables que 1h
    
    # ⚠️ COSTOS DE TRANSACCIÓN REALISTAS
    COMISION_MAKER: float = 0.001      # 0.1% Binance Maker
    COMISION_TAKER: float = 0.001      # 0.1% Binance Taker
    SLIPPAGE_PROMEDIO: float = 0.0005  # 0.05% slippage promedio
    LATENCIA_MS: int = 100             # 100ms latencia ejecución
    
    # Gestión de riesgo (más conservadora)
    STOP_LOSS_PCT: float = 0.02        # 2%
    TAKE_PROFIT_PCT: float = 0.04      # 4%
    RATIO_MINIMO_RR: float = 1.8       # Mínimo 1.8:1
    MAX_RIESGO_POR_OPERACION: float = 0.01  # 1% max por operación
    MAX_OPERACIONES_DIARIAS: int = 3   # Límite diario
    
    # Validación (más estricta)
    N_FOLDS_WF: int = 4
    MIN_MUESTRAS_ENTRENAMIENTO: int = 500
    MIN_MUESTRAS_CLASE: int = 30
    
    # Umbrales (más realistas)
    UMBRAL_PROBABILIDAD_MIN: float = 0.58  # 58% (no 52%)
    UMBRAL_CONFIANZA_MIN: float = 0.60     # 60% confianza
    UMBRAL_MOVIMIENTO: float = 0.015       # 1.5% movimiento mínimo
    
    # Filtros RSI (más estrictos)
    RSI_EXTREME_LOW: int = 20   # No operar bajo 20
    RSI_EXTREME_HIGH: int = 80  # No operar sobre 80
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    # Validación estadística
    P_VALUE_MAX: float = 0.05          # Significancia estadística
    MIN_SHARPE_RATIO: float = 1.0      # Sharpe mínimo
    MIN_PROFIT_FACTOR: float = 1.5     # PF mínimo
    MIN_WIN_RATE: float = 0.48         # Win rate mínimo
    MAX_DRAWDOWN: float = 0.15         # Max DD aceptable
    
    # Paths
    MODELOS_DIR: Path = Path("modelos_trading_pro")
    DB_PATH: Path = Path("trading_data.db")
    
    @classmethod
    def get_fechas(cls):
        """Calcula fechas respetando límites de Yahoo"""
        now = datetime.now(cls.TIMEZONE)
        fecha_max_retroceso = now - timedelta(days=730)
        
        inicio_backtest = now - timedelta(days=cls.DIAS_BACKTEST)
        inicio_validacion = inicio_backtest - timedelta(days=cls.DIAS_VALIDACION)
        inicio_entrenamiento = inicio_validacion - timedelta(days=cls.DIAS_ENTRENAMIENTO)
        
        if inicio_entrenamiento < fecha_max_retroceso:
            logger.warning("⚠️ Ajustando fechas por límite de Yahoo Finance")
            inicio_entrenamiento = fecha_max_retroceso
        
        return {
            'actual': now,
            'inicio_entrenamiento': inicio_entrenamiento,
            'inicio_validacion': inicio_validacion,
            'inicio_backtest': inicio_backtest,
            'fecha_max_retroceso': fecha_max_retroceso
        }
    
    def validate(self) -> bool:
        """Valida configuración"""
        assert self.RATIO_MINIMO_RR >= 1.5, "R:R debe ser >= 1.5"
        assert self.UMBRAL_PROBABILIDAD_MIN > 0.55, "Probabilidad mín debe ser > 55%"
        assert self.MIN_SHARPE_RATIO >= 0.5, "Sharpe debe ser >= 0.5"
        assert len(self.ACTIVOS) > 0, "Debe haber al menos 1 activo"
        return True


# ============================================
# BASE DE DATOS
# ============================================

class TradingDatabase:
    """Gestión de persistencia con SQLite"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexiones"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en DB: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Crea tablas si no existen"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabla de operaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operaciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TIMESTAMP,
                    ticker TEXT,
                    direccion TEXT,
                    precio_entrada REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    ratio_rr REAL,
                    probabilidad REAL,
                    confianza REAL,
                    resultado TEXT,
                    retorno REAL,
                    comision REAL,
                    retorno_neto REAL,
                    velas_hasta_cierre INTEGER,
                    modo TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de métricas de modelo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metricas_modelo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TIMESTAMP,
                    ticker TEXT,
                    horizonte INTEGER,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    sharpe_ratio REAL,
                    profit_factor REAL,
                    win_rate REAL,
                    max_drawdown REAL,
                    n_operaciones INTEGER,
                    retorno_total REAL,
                    p_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de señales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS senales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TIMESTAMP,
                    ticker TEXT,
                    senal TEXT,
                    precio REAL,
                    probabilidad REAL,
                    confianza REAL,
                    fuerza TEXT,
                    enviada BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("✅ Base de datos inicializada")
    
    def guardar_operacion(self, operacion: Dict):
        """Guarda operación en DB"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO operaciones (
                    fecha, ticker, direccion, precio_entrada, stop_loss, 
                    take_profit, ratio_rr, probabilidad, confianza, resultado, 
                    retorno, comision, retorno_neto, velas_hasta_cierre, modo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operacion['fecha'],
                operacion['ticker'],
                operacion['direccion'],
                operacion['precio_entrada'],
                operacion['stop_loss'],
                operacion['take_profit'],
                operacion['ratio_rr'],
                operacion.get('probabilidad', 0),
                operacion.get('confianza', 0),
                operacion['resultado'],
                operacion['retorno'],
                operacion.get('comision', 0),
                operacion.get('retorno_neto', 0),
                operacion['velas_hasta_cierre'],
                operacion.get('modo', 'backtest')
            ))
    
    def guardar_metricas(self, ticker: str, horizonte: int, metricas: Dict):
        """Guarda métricas de modelo"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metricas_modelo (
                    fecha, ticker, horizonte, accuracy, precision_score, 
                    recall_score, f1_score, sharpe_ratio, profit_factor, 
                    win_rate, max_drawdown, n_operaciones, retorno_total, p_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                ticker,
                horizonte,
                metricas.get('accuracy', 0),
                metricas.get('precision', 0),
                metricas.get('recall', 0),
                metricas.get('f1', 0),
                metricas.get('sharpe_ratio', 0),
                metricas.get('profit_factor', 0),
                metricas.get('tasa_exito', 0),
                metricas.get('max_drawdown', 0),
                metricas.get('n_operaciones', 0),
                metricas.get('retorno_total', 0),
                metricas.get('p_value', 1.0)
            ))
    
    def obtener_ultimas_operaciones(self, ticker: str, limit: int = 10) -> pd.DataFrame:
        """Obtiene últimas operaciones de un ticker"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM operaciones 
                WHERE ticker = ? 
                ORDER BY fecha DESC 
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(ticker, limit))


# ============================================
# DESCARGA DE DATOS
# ============================================

class YahooDataDownloader:
    """Descarga inteligente con reintentos y validación"""
    
    @staticmethod
    def validar_datos(df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
        """Valida calidad de datos descargados"""
        if df.empty:
            return False, "DataFrame vacío"
        
        # Verificar columnas necesarias
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return False, f"Faltan columnas: {missing}"
        
        # Verificar valores nulos
        null_pct = df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))
        if null_pct > 0.05:  # Más de 5% nulos
            return False, f"Demasiados valores nulos: {null_pct:.1%}"
        
        # Verificar precios válidos
        if (df['Close'] <= 0).any():
            return False, "Precios negativos o cero detectados"
        
        # Verificar High >= Low
        if (df['High'] < df['Low']).any():
            return False, "High < Low detectado"
        
        # Verificar gaps extremos (>50% en una vela)
        returns = df['Close'].pct_change().abs()
        if (returns > 0.5).any():
            logger.warning(f"⚠️ {ticker}: Gaps extremos detectados (>50%)")
        
        # Verificar continuidad temporal
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_diff * 2]  # Gaps > 2 horas
        
        if len(gaps) > len(df) * 0.1:  # Más de 10% con gaps
            logger.warning(f"⚠️ {ticker}: Muchos gaps temporales ({len(gaps)})")
        
        return True, "OK"
    
    @staticmethod
    def descargar_con_reintentos(
        ticker: str, 
        fecha_inicio: datetime, 
        fecha_fin: datetime,
        intervalo: str = "1h", 
        max_reintentos: int = 3
    ) -> pd.DataFrame:
        """Descarga con reintentos y validación"""
        
        for intento in range(max_reintentos):
            try:
                logger.info(f"📥 Descargando {ticker} (intento {intento+1}/{max_reintentos})...")
                
                # Pequeña pausa entre reintentos
                if intento > 0:
                    time.sleep(2 ** intento)
                
                df = yf.download(
                    ticker,
                    start=fecha_inicio,
                    end=fecha_fin,
                    interval=intervalo,
                    progress=False,
                    threads=False,
                    auto_adjust=True  # Ajustar por splits/dividendos
                )
                
                if df.empty:
                    logger.warning(f"  ⚠️ Descarga vacía para {ticker}")
                    continue
                
                # Normalizar columnas
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Asegurar columnas necesarias
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df[required].copy()
                
                # Limpiar datos
                df = df.dropna()
                df = df[df['Close'] > 0]
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
                
                # Validar calidad
                is_valid, msg = YahooDataDownloader.validar_datos(df, ticker)
                if not is_valid:
                    logger.error(f"  ❌ Validación falló: {msg}")
                    continue
                
                logger.info(f"  ✅ {len(df)} velas descargadas ({df.index[0].date()} a {df.index[-1].date()})")
                return df
                
            except Exception as e:
                logger.error(f"  ❌ Error descargando {ticker}: {e}")
                if intento == max_reintentos - 1:
                    logger.error(f"  ❌ Todos los intentos fallaron para {ticker}")
        
        return pd.DataFrame()


# ============================================
# INDICADORES TÉCNICOS
# ============================================

class IndicadoresTecnicos:
    """Features técnicas sin look-ahead bias"""
    
    @staticmethod
    def calcular_rsi(precios: pd.Series, periodo: int = 14) -> pd.Series:
        """RSI sin look-ahead"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo, min_periods=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo, min_periods=periodo).mean()
        
        # Evitar división por cero
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    @staticmethod
    def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
        """ATR sin look-ahead"""
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=periodo, min_periods=periodo).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ CRÍTICO: Todas las features usan .shift(1) para evitar look-ahead
        """
        df = df.copy()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # === RETORNOS PASADOS ===
        for periodo in [1, 4, 8, 12, 24]:
            df[f'retorno_{periodo}h'] = close.pct_change(periodo).shift(1)
        
        # === VOLATILIDAD PASADA ===
        retornos = close.pct_change(1)
        df['volatilidad_6h'] = retornos.rolling(6).std().shift(1)
        df['volatilidad_12h'] = retornos.rolling(12).std().shift(1)
        df['volatilidad_24h'] = retornos.rolling(24).std().shift(1)
        
        # === RSI PASADO ===
        for periodo in [14, 21]:  # Reducir redundancia
            rsi_raw = IndicadoresTecnicos.calcular_rsi(close, periodo)
            df[f'RSI_{periodo}'] = rsi_raw.shift(1)
        
        # === MEDIAS MÓVILES PASADAS ===
        for periodo in [12, 24, 50]:
            sma = close.rolling(periodo, min_periods=periodo).mean().shift(1)
            df[f'SMA_{periodo}'] = sma
            df[f'dist_sma_{periodo}'] = ((close.shift(1) - sma) / sma)
        
        # === EMA PASADAS ===
        ema_12 = close.ewm(span=12, adjust=False).mean().shift(1)
        ema_26 = close.ewm(span=26, adjust=False).mean().shift(1)
        df['EMA_12'] = ema_12
        df['EMA_26'] = ema_26
        df['MACD'] = ema_12 - ema_26
        
        # === ATR PASADO ===
        atr = IndicadoresTecnicos.calcular_atr(df, 14)
        df['ATR'] = atr.shift(1)
        df['ATR_pct'] = (atr / close).shift(1)
        
        # === VOLUMEN RELATIVO ===
        vol_ma_24 = volume.rolling(24, min_periods=24).mean()
        df['volumen_rel'] = (volume / vol_ma_24).shift(1)
        
        # === RANGO HL ===
        rango = (high - low) / close
        df['rango_hl_pct'] = rango.shift(1)
        df['rango_hl_ma'] = rango.rolling(12).mean().shift(1)
        
        # === TENDENCIA ===
        df['tendencia'] = (df['SMA_12'] > df['SMA_24']).astype(int)
        
        # === MOMENTUM ===
        df['momentum_12h'] = (close / close.shift(12) - 1).shift(1)
        df['momentum_24h'] = (close / close.shift(24) - 1).shift(1)
        
        # === BOLLINGER BANDS ===
        sma_20 = close.rolling(20, min_periods=20).mean().shift(1)
        std_20 = close.rolling(20, min_periods=20).std().shift(1)
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (close.shift(1) - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === VOLATILIDAD RELATIVA ===
        df['vol_ratio'] = df['volatilidad_6h'] / df['volatilidad_24h']
        
        return df


# ============================================
# ETIQUETADO
# ============================================

class EtiquetadoDatos:
    """Etiquetado con umbrales dinámicos"""
    
    @staticmethod
    def crear_etiquetas_direccion(df: pd.DataFrame, horizonte: int) -> Tuple[pd.Series, pd.Series]:
        """
        Etiquetado con umbral dinámico basado en volatilidad
        
        Returns:
            (etiquetas, retornos_futuros)
        """
        # Retorno futuro
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        # Umbral dinámico basado en volatilidad reciente
        volatilidad = df['Close'].pct_change().rolling(24).std().fillna(0.015)
        umbral_dinamico = volatilidad * 2.0  # 2x la volatilidad
        
        # Limitar umbral entre 1% y 2.5%
        umbral_usado = umbral_dinamico.clip(lower=0.01, upper=0.025)
        
        # Crear etiquetas
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > umbral_usado] = 1   # LONG
        etiqueta[retorno_futuro < -umbral_usado] = 0  # SHORT
        # Entre umbrales = NaN (no operar - zona neutral)
        
        # Logging de distribución
        dist = etiqueta.value_counts()
        total = len(etiqueta.dropna())
        if total > 0:
            logger.info(f"    Etiquetas {horizonte}h: LONG={dist.get(1,0)} ({dist.get(1,0)/total:.1%}), "
                       f"SHORT={dist.get(0,0)} ({dist.get(0,0)/total:.1%}), "
                       f"NEUTRAL={etiqueta.isna().sum()} ({etiqueta.isna().sum()/len(etiqueta):.1%})")
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df: pd.DataFrame, horizonte: int) -> Tuple[pd.DataFrame, List[str]]:
        """Prepara dataset completo para ML"""
        
        # Calcular features
        df = IndicadoresTecnicos.calcular_features(df)
        
        # Crear etiquetas
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Lista de features (sin redundancia excesiva)
        features_base = [
            # RSI
            'RSI_14', 'RSI_21',
            # Volatilidad
            'volatilidad_12h', 'volatilidad_24h', 'vol_ratio',
            # Distancias a medias
            'dist_sma_12', 'dist_sma_24', 'dist_sma_50',
            # MACD
            'MACD',
            # ATR
            'ATR_pct',
            # Tendencia
            'tendencia',
            # Retornos
            'retorno_1h', 'retorno_4h', 'retorno_8h', 'retorno_24h',
            # Volumen
            'volumen_rel',
            # Rango
            'rango_hl_pct', 'rango_hl_ma',
            # Momentum
            'momentum_12h', 'momentum_24h',
            # Bollinger
            'bb_position'
        ]
        
        # Filtrar solo features disponibles
        features_disponibles = [f for f in features_base if f in df.columns]
        
        logger.info(f"    Features disponibles: {len(features_disponibles)}")
        
        return df, features_disponibles


# ============================================
# MODELO CON VALIDACIÓN ESTADÍSTICA
# ============================================

class ModeloPrediccion:
    """Modelo con validación robusta"""
    
    def __init__(self, horizonte: int, ticker: str, config: TradingConfig):
        self.horizonte = horizonte
        self.ticker = ticker
        self.config = config
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
        self.feature_importance = {}
        self.is_valid = False
    
    def seleccionar_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        max_features: int = 15
    ) -> List[str]:
        """
        ✅ Selección de features SOLO con datos de entrenamiento
        """
        from sklearn.feature_selection import mutual_info_classif
        
        # Calcular mutual information
        mi_scores = mutual_info_classif(
            X_train.fillna(0), 
            y_train, 
            random_state=42
        )
        
        # Crear DataFrame de scores
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': mi_scores
        }).sort_values('score', ascending=False)
        
        # Seleccionar top features
        selected = feature_scores.head(max_features)['feature'].tolist()
        
        logger.info(f"    Top 5 features seleccionadas:")
        for idx, row in feature_scores.head(5).iterrows():
            logger.info(f"      {row['feature']}: {row['score']:.4f}")
        
        return selected
    
    def validar_estadisticamente(
        self, 
        retornos: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        ✅ Validación estadística robusta
        """
        # Test t de Student: ¿retornos significativamente > 0?
        t_stat, p_value = stats.ttest_1samp(retornos, 0)
        
        # Sharpe ratio
        sharpe = retornos.mean() / retornos.std() * np.sqrt(365*24) if retornos.std() > 0 else 0
        
        # Profit factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        profit_factor = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Win rate
        win_rate = (retornos > 0).sum() / len(retornos)
        
        # Max drawdown
        equity_curve = (1 + retornos).cumprod()
        drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        resultados = {
            'p_value': p_value,
            't_statistic': t_stat,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': drawdown,
            'retorno_medio': retornos.mean(),
            'retorno_std': retornos.std()
        }
        
        # Criterios de aceptación
        criterios = [
            p_value < self.config.P_VALUE_MAX,
            sharpe >= self.config.MIN_SHARPE_RATIO,
            profit_factor >= self.config.MIN_PROFIT_FACTOR,
            win_rate >= self.config.MIN_WIN_RATE,
            abs(drawdown) <= self.config.MAX_DRAWDOWN
        ]
        
        es_valido = sum(criterios) >= 4  # Al menos 4 de 5 criterios
        
        logger.info(f"    Validación estadística:")
        logger.info(f"      {'✅' if criterios[0] else '❌'} p-value: {p_value:.4f} (< {self.config.P_VALUE_MAX})")
        logger.info(f"      {'✅' if criterios[1] else '❌'} Sharpe: {sharpe:.2f} (>= {self.config.MIN_SHARPE_RATIO})")
        logger.info(f"      {'✅' if criterios[2] else '❌'} Profit Factor: {profit_factor:.2f} (>= {self.config.MIN_PROFIT_FACTOR})")
        logger.info(f"      {'✅' if criterios[3] else '❌'} Win Rate: {win_rate:.2%} (>= {self.config.MIN_WIN_RATE:.0%})")
        logger.info(f"      {'✅' if criterios[4] else '❌'} Max DD: {drawdown:.2%} (<= {self.config.MAX_DRAWDOWN:.0%})")
        
        return es_valido, resultados
    
    def entrenar_walk_forward(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        etiqueta_col: str
    ) -> bool:
        """
        ✅ Walk-forward validation con validación estadística
        """
        # Limpiar datos
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < self.config.MIN_MUESTRAS_ENTRENAMIENTO:
            logger.warning(f"    ⚠️ Datos insuficientes: {len(df_valido)}")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            logger.warning(f"    ⚠️ Solo una clase presente")
            return False
        
        if class_counts.min() < self.config.MIN_MUESTRAS_CLASE:
            logger.warning(f"    ⚠️ Clase minoritaria insuficiente: {class_counts.min()}")
            return False
        
        logger.info(f"    Muestras: {len(X)} | LONG: {class_counts.get(1,0)} | SHORT: {class_counts.get(0,0)}")
        
        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.N_FOLDS_WF,
            test_size=max(200, len(X) // 10),
            gap=self.horizonte * 2
        )
        
        fold_scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Verificar clases en fold
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                logger.warning(f"      Fold {fold}: Clases insuficientes, saltando...")
                continue
            
            # Seleccionar features (solo con datos de entrenamiento)
            if fold == 1:
                self.features = self.seleccionar_features(
                    X_train[features], 
                    y_train, 
                    max_features=15
                )
            
            X_train_sel = X_train[self.features]
            X_val_sel = X_val[self.features]
            
            # Escalar
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_val_scaled = scaler.transform(X_val_sel)
            
            # Modelo
            modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_val_scaled)
            y_proba = modelo.predict_proba(X_val_scaled)
            
            # Métricas
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0, average='binary')
            rec = recall_score(y_val, y_pred, zero_division=0, average='binary')
            f1 = f1_score(y_val, y_pred, zero_division=0, average='binary')
            
            fold_scores.append({
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
            
            # Guardar predicciones para validación estadística
            fold_predictions.append({
                'y_true': y_val,
                'y_pred': y_pred,
                'y_proba': y_proba
            })
            
            logger.info(f"      Fold {fold}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
        
        if len(fold_scores) == 0:
            logger.warning("    ⚠️ No se completó ningún fold")
            return False
        
        # Métricas promedio
        self.metricas_validacion = {
            'accuracy': np.mean([s['accuracy'] for s in fold_scores]),
            'precision': np.mean([s['precision'] for s in fold_scores]),
            'recall': np.mean([s['recall'] for s in fold_scores]),
            'f1': np.mean([s['f1'] for s in fold_scores]),
            'std_accuracy': np.std([s['accuracy'] for s in fold_scores]),
            'n_folds': len(fold_scores)
        }
        
        logger.info(f"    Métricas promedio: Acc={self.metricas_validacion['accuracy']:.3f} "
                   f"Prec={self.metricas_validacion['precision']:.3f} "
                   f"Rec={self.metricas_validacion['recall']:.3f} "
                   f"F1={self.metricas_validacion['f1']:.3f}")
        
        # Criterio mínimo de accuracy
        if self.metricas_validacion['accuracy'] < 0.52:
            logger.warning(f"    ❌ Accuracy insuficiente: {self.metricas_validacion['accuracy']:.2%}")
            return False
        
        # Entrenar modelo final con todos los datos
        logger.info("    Entrenando modelo final...")
        
        X_final = X[self.features]
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_final)
        
        self.modelo = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=7,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.modelo.fit(X_scaled, y)
        
        # Feature importance
        self.feature_importance = dict(zip(self.features, self.modelo.feature_importances_))
        
        logger.info("    ✅ Modelo entrenado exitosamente")
        return True
    
    def predecir(self, df_actual: pd.DataFrame) -> Optional[Dict]:
        """Predicción en tiempo real"""
        if self.modelo is None or not self.is_valid:
            return None
        
        if not all(f in df_actual.columns for f in self.features):
            logger.warning(f"    Faltan features para predicción")
            return None
        
        X = df_actual[self.features].iloc[[-1]]
        
        if X.isnull().any().any():
            return None
        
        try:
            X_scaled = self.scaler.transform(X)
            
            prediccion_clase = self.modelo.predict(X_scaled)[0]
            probabilidades = self.modelo.predict_proba(X_scaled)[0]
            
            return {
                'prediccion': int(prediccion_clase),
                'probabilidad_positiva': probabilidades[1],
                'probabilidad_negativa': probabilidades[0],
                'confianza': max(probabilidades)
            }
        except Exception as e:
            logger.error(f"    Error en predicción: {e}")
            return None
    
    def guardar(self, path: Path) -> bool:
        """Guarda modelo"""
        if self.modelo is None:
            return False
        
        try:
            modelo_data = {
                'modelo': self.modelo,
                'scaler': self.scaler,
                'features': self.features,
                'metricas': self.metricas_validacion,
                'feature_importance': self.feature_importance,
                'horizonte': self.horizonte,
                'ticker': self.ticker,
                'is_valid': self.is_valid,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo_data, path)
            logger.info(f"    💾 Modelo guardado: {path}")
            return True
        except Exception as e:
            logger.error(f"    Error guardando modelo: {e}")
            return False
    
    @classmethod
    def cargar(cls, path: Path, config: TradingConfig) -> 'ModeloPrediccion':
        """Carga modelo"""
        modelo_data = joblib.load(path)
        
        instancia = cls(modelo_data['horizonte'], modelo_data['ticker'], config)
        instancia.modelo = modelo_data['modelo']
        instancia.scaler = modelo_data['scaler']
        instancia.features = modelo_data['features']
        instancia.metricas_validacion = modelo_data['metricas']
        instancia.feature_importance = modelo_data.get('feature_importance', {})
        instancia.is_valid = modelo_data.get('is_valid', False)
        
        logger.info(f"    📂 Modelo cargado: {path}")
        return instancia


# ============================================
# BACKTESTING REALISTA
# ============================================

class Backtester:
    """Backtesting con costos y slippage realistas"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        modelos: Dict[int, ModeloPrediccion], 
        ticker: str,
        config: TradingConfig,
        db: TradingDatabase
    ):
        self.df = df
        self.modelos = modelos
        self.ticker = ticker
        self.config = config
        self.db = db
        self.operaciones = []
    
    def calcular_costos(
        self, 
        precio_entrada: float, 
        precio_salida: float,
        es_maker: bool = True
    ) -> float:
        """
        Calcula costos totales de transacción
        """
        comision = self.config.COMISION_MAKER if es_maker else self.config.COMISION_TAKER
        
        # Comisión entrada + salida
        costo_comision = comision * 2
        
        # Slippage
        costo_slippage = self.config.SLIPPAGE_PROMEDIO * 2
        
        return costo_comision + costo_slippage
    
    def simular_operacion(
        self, 
        idx: pd.Timestamp, 
        señal_long: bool, 
        prob: float,
        confianza: float
    ) -> Optional[Dict]:
        """
        Simula operación individual con costos realistas
        """
        precio_entrada = self.df.loc[idx, 'Close']
        
        # Aplicar slippage en entrada
        if señal_long:
            precio_entrada *= (1 + self.config.SLIPPAGE_PROMEDIO)
        else:
            precio_entrada *= (1 - self.config.SLIPPAGE_PROMEDIO)
        
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Calcular niveles
        if señal_long:
            stop_loss = precio_entrada * (1 - self.config.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 + self.config.TAKE_PROFIT_PCT)
        else:
            stop_loss = precio_entrada * (1 + self.config.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 - self.config.TAKE_PROFIT_PCT)
        
        ratio_rr = abs(take_profit - precio_entrada) / abs(precio_entrada - stop_loss)
        
        if ratio_rr < self.config.RATIO_MINIMO_RR:
            return None
        
        # Simular hasta 24 horas
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(24, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]
        
        resultado = 'TIEMPO'
        velas_hasta_cierre = max_ventana
        precio_salida = precios_futuros.iloc[-1]['Close']
        
        # Simular ejecución vela por vela
        for i in range(1, len(precios_futuros)):
            high = precios_futuros.iloc[i]['High']
            low = precios_futuros.iloc[i]['Low']
            
            if señal_long:
                # Verificar SL primero (más conservador)
                if low <= stop_loss:
                    resultado = 'SL'
                    precio_salida = stop_loss
                    velas_hasta_cierre = i
                    break
                elif high >= take_profit:
                    resultado = 'TP'
                    precio_salida = take_profit
                    velas_hasta_cierre = i
                    break
            else:
                if high >= stop_loss:
                    resultado = 'SL'
                    precio_salida = stop_loss
                    velas_hasta_cierre = i
                    break
                elif low <= take_profit:
                    resultado = 'TP'
                    precio_salida = take_profit
                    velas_hasta_cierre = i
                    break
        
        # Si termina por tiempo
        if resultado == 'TIEMPO':
            precio_salida = precios_futuros.iloc[velas_hasta_cierre]['Close']
            # Aplicar slippage en salida
            if señal_long:
                precio_salida *= (1 - self.config.SLIPPAGE_PROMEDIO)
            else:
                precio_salida *= (1 + self.config.SLIPPAGE_PROMEDIO)
        
        # Calcular retorno BRUTO
        if señal_long:
            retorno_bruto = (precio_salida - precio_entrada) / precio_entrada
        else:
            retorno_bruto = (precio_entrada - precio_salida) / precio_entrada
        
        # Calcular costos
        costos_totales = self.calcular_costos(precio_entrada, precio_salida)
        
        # Retorno NETO
        retorno_neto = retorno_bruto - costos_totales
        
        operacion = {
            'fecha': idx,
            'ticker': self.ticker,
            'direccion': direccion,
            'precio_entrada': precio_entrada,
            'precio_salida': precio_salida,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ratio_rr': ratio_rr,
            'probabilidad': prob,
            'confianza': confianza,
            'resultado': resultado,
            'retorno': retorno_bruto,
            'comision': costos_totales,
            'retorno_neto': retorno_neto,
            'velas_hasta_cierre': velas_hasta_cierre
        }
        
        return operacion
    
    def ejecutar(self, fecha_inicio: pd.Timestamp) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Ejecuta backtest completo"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            logger.warning(f"  ⚠️ Datos insuficientes para backtest")
            return None
        
        logger.info(f"  📊 Periodo: {df_backtest.index[0].date()} a {df_backtest.index[-1].date()}")
        logger.info(f"  📈 Velas: {len(df_backtest)}")
        
        operaciones_dia = {}
        
        for idx in df_backtest.index[:-24]:
            # Límite de operaciones diarias
            fecha_dia = idx.date()
            if operaciones_dia.get(fecha_dia, 0) >= self.config.MAX_OPERACIONES_DIARIAS:
                continue
            
            # Obtener predicciones de todos los modelos
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                if not modelo.is_valid:
                    continue
                
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Promediar probabilidades
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # Filtros de confianza
            if confianza_promedio < self.config.UMBRAL_CONFIANZA_MIN:
                continue
            
            # Determinar señal
            if prob_promedio > 0.5:
                señal_long = True
                prob_real = prob_promedio
            else:
                señal_long = False
                prob_real = 1 - prob_promedio
            
            # Filtro de probabilidad
            if prob_real < self.config.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            # Filtros RSI
            rsi = df_backtest.loc[idx, 'RSI_14'] if 'RSI_14' in df_backtest.columns else 50
            
            if señal_long and rsi > self.config.RSI_EXTREME_HIGH:
                continue
            
            if not señal_long and rsi < self.config.RSI_EXTREME_LOW:
                continue
            
            # Simular operación
            operacion = self.simular_operacion(idx, señal_long, prob_real, confianza_promedio)
            
            if operacion:
                self.operaciones.append(operacion)
                operaciones_dia[fecha_dia] = operaciones_dia.get(fecha_dia, 0) + 1
                
                # Guardar en DB
                self.db.guardar_operacion(operacion)
        
        if not self.operaciones:
            logger.warning(f"  ⚠️ No se generaron operaciones")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self) -> Tuple[Dict, pd.DataFrame]:
        """Calcula métricas de rendimiento"""
        df_ops = pd.DataFrame(self.operaciones)
        
        # Métricas básicas
        n_ops = len(df_ops)
        n_tp = (df_ops['resultado'] == 'TP').sum()
        n_sl = (df_ops['resultado'] == 'SL').sum()
        n_timeout = (df_ops['resultado'] == 'TIEMPO').sum()
        
        # Usar retornos NETOS
        retornos_netos = df_ops['retorno_neto']
        operaciones_ganadoras = retornos_netos > 0
        
        # Profit factor
        ganancias = retornos_netos[retornos_netos > 0].sum()
        perdidas = abs(retornos_netos[retornos_netos < 0].sum())
        profit_factor = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Equity curve
        equity_curve = (1 + retornos_netos).cumprod()
        
        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        sharpe_ratio = retornos_netos.mean() / retornos_netos.std() * np.sqrt(365*24) if retornos_netos.std() > 0 else 0
        
        # Test estadístico
        t_stat, p_value = stats.ttest_1samp(retornos_netos, 0)
        
        # Costos totales
        costos_totales = df_ops['comision'].sum()
        
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': operaciones_ganadoras.sum() / n_ops,
            'hit_tp_rate': n_tp / n_ops,
            'hit_sl_rate': n_sl / n_ops,
            'timeout_rate': n_timeout / n_ops,
            'retorno_total_bruto': df_ops['retorno'].sum(),
            'retorno_total_neto': retornos_netos.sum(),
            'retorno_promedio': retornos_netos.mean(),
            'retorno_mediano': retornos_netos.median(),
            'mejor_operacion': retornos_netos.max(),
            'peor_operacion': retornos_netos.min(),
            'promedio_ganador': retornos_netos[retornos_netos > 0].mean() if (retornos_netos > 0).any() else 0,
            'promedio_perdedor': retornos_netos[retornos_netos < 0].mean() if (retornos_netos < 0).any() else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_final': equity_curve.iloc[-1] if len(equity_curve) > 0 else 1,
            'duracion_promedio': df_ops['velas_hasta_cierre'].mean(),
            'costos_totales': costos_totales,
            'p_value': p_value,
            't_statistic': t_stat
        }
        
        return metricas, df_ops


# ============================================
# SISTEMA PRINCIPAL
# ============================================

class SistemaTradingTicker:
    """Sistema completo para un ticker"""
    
    def __init__(self, ticker: str, config: TradingConfig, db: TradingDatabase):
        self.ticker = ticker
        self.config = config
        self.db = db
        self.modelos = {}
        self.fechas = config.get_fechas()
        self.df_historico = None
        self.metricas_backtest = None
    
    def descargar_datos(self) -> bool:
        """Descarga datos históricos"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📥 DESCARGANDO {self.ticker}")
        logger.info(f"{'='*80}")
        
        try:
            df = YahooDataDownloader.descargar_con_reintentos(
                self.ticker,
                self.fechas['inicio_entrenamiento'],
                self.fechas['actual'],
                self.config.INTERVALO
            )
            
            if df.empty:
                logger.error(f"  ❌ No se pudieron descargar datos")
                return False
            
            self.df_historico = df
            logger.info(f"  ✅ {len(df)} velas descargadas")
            logger.info(f"     Periodo: {df.index[0].date()} → {df.index[-1].date()}")
            logger.info(f"     Precio actual: ${df['Close'].iloc[-1]:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error descargando datos: {e}")
            return False
    
    def entrenar_modelos(self) -> bool:
        """Entrena modelos para cada horizonte"""
        logger.info(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        logger.info("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Datos hasta inicio de backtest
        df_train = self.df_historico[
            self.df_historico.index < self.fechas['inicio_backtest']
        ].copy()
        
        logger.info(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        logger.info(f"  📅 Periodo: {df_train.index[0].date()} → {df_train.index[-1].date()}")
        
        modelos_entrenados = 0
        
        for horizonte in self.config.HORIZONTES:
            logger.info(f"\n  🔄 Horizonte {horizonte}h...")
            
            try:
                df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
                etiqueta_col = f'etiqueta_{horizonte}h'
                
                modelo = ModeloPrediccion(horizonte, self.ticker, self.config)
                
                if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                    # Validar con backtest interno
                    logger.info("    Validando con backtest...")
                    
                    df_val = self.df_historico[
                        (self.df_historico.index >= self.fechas['inicio_validacion']) &
                        (self.df_historico.index < self.fechas['inicio_backtest'])
                    ].copy()
                    
                    if len(df_val) > 100:
                        df_val_prep, _ = EtiquetadoDatos.preparar_dataset_ml(df_val, horizonte)
                        
                        backtester_val = Backtester(
                            df_val_prep, 
                            {horizonte: modelo}, 
                            self.ticker,
                            self.config,
                            self.db
                        )
                        
                        resultado_val = backtester_val.ejecutar(self.fechas['inicio_validacion'])
                        
                        if resultado_val:
                            metricas_val, _ = resultado_val
                            
                            # Validación estadística
                            retornos_val = pd.DataFrame(backtester_val.operaciones)['retorno_neto'].values
                            
                            es_valido, stats_val = modelo.validar_estadisticamente(retornos_val)
                            
                            if es_valido:
                                modelo.is_valid = True
                                self.modelos[horizonte] = modelo
                                modelos_entrenados += 1
                                
                                # Guardar métricas en DB
                                self.db.guardar_metricas(
                                    self.ticker, 
                                    horizonte, 
                                    {**metricas_val, **stats_val}
                                )
                                
                                logger.info(f"    ✅ Modelo validado y aceptado")
                            else:
                                logger.warning(f"    ❌ Modelo no pasó validación estadística")
                        else:
                            logger.warning(f"    ⚠️ Validación falló (sin operaciones)")
                    
            except Exception as e:
                logger.error(f"    ❌ Error entrenando: {e}", exc_info=True)
                continue
        
        logger.info(f"\n  ✅ Modelos válidos: {modelos_entrenados}/{len(self.config.HORIZONTES)}")
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self) -> bool:
        """Ejecuta backtest final"""
        logger.info(f"\n🔬 BACKTESTING FINAL - {self.ticker}")
        logger.info("-" * 80)
        
        if not self.modelos:
            logger.error("  ❌ No hay modelos válidos")
            return False
        
        # Preparar datos completos
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(self.df_historico, 1)
        
        backtester = Backtester(
            df_completo, 
            self.modelos, 
            self.ticker,
            self.config,
            self.db
        )
        
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        logger.info(f"\n  📊 RESULTADOS:")
        logger.info(f"    Operaciones: {metricas['n_operaciones']}")
        logger.info(f"    Win rate: {metricas['tasa_exito']:.2%}")
        logger.info(f"    Retorno bruto: {metricas['retorno_total_bruto']:.2%}")
        logger.info(f"    Retorno NETO: {metricas['retorno_total_neto']:.2%}")
        logger.info(f"    Costos totales: {metricas['costos_totales']:.2%}")
        logger.info(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        logger.info(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        logger.info(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        logger.info(f"    p-value: {metricas['p_value']:.4f}")
        
        if len(df_ops) > 0:
            long_ops = df_ops[df_ops['direccion'] == 'LONG']
            short_ops = df_ops[df_ops['direccion'] == 'SHORT']
            
            if len(long_ops) > 0:
                wr_long = (long_ops['retorno_neto'] > 0).sum() / len(long_ops)
                logger.info(f"    LONG: {len(long_ops)} ops, Win rate: {wr_long:.1%}")
            
            if len(short_ops) > 0:
                wr_short = (short_ops['retorno_neto'] > 0).sum() / len(short_ops)
                logger.info(f"    SHORT: {len(short_ops)} ops, Win rate: {wr_short:.1%}")
        
        return True
    
    def es_viable(self) -> Tuple[bool, int]:
        """Evalúa viabilidad del sistema"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        
        criterios = [
            ('Operaciones >= 10', m['n_operaciones'] >= 10),
            ('Win rate > 45%', m['tasa_exito'] > 0.45),
            ('Profit factor > 1.5', m['profit_factor'] > 1.5),
            ('Retorno neto > 0%', m['retorno_total_neto'] > 0),
            ('Sharpe > 1.0', m['sharpe_ratio'] > 1.0),
            ('Max DD < 15%', abs(m['max_drawdown']) < 0.15),
            ('p-value < 0.05', m['p_value'] < 0.05)
        ]
        
        cumplidos = sum([c[1] for c in criterios])
        viable = cumplidos >= 5  # Al menos 5 de 7
        
        logger.info(f"\n  📋 EVALUACIÓN:")
        for nombre, resultado in criterios:
            logger.info(f"    {'✅' if resultado else '❌'} {nombre}")
        
        return viable, cumplidos
    
    def guardar_modelos(self) -> bool:
        """Guarda modelos entrenados"""
        if not self.modelos:
            return False
        
        path_ticker = self.config.MODELOS_DIR / self.ticker
        path_ticker.mkdir(parents=True, exist_ok=True)
        
        for horizonte, modelo in self.modelos.items():
            if modelo.is_valid:
                path_modelo = path_ticker / f"modelo_{horizonte}h.pkl"
                modelo.guardar(path_modelo)
        
        if self.metricas_backtest:
            path_metricas = path_ticker / "metricas_backtest.json"
            with open(path_metricas, 'w') as f:
                # Convertir numpy types a Python types
                metricas_json = {}
                for k, v in self.metricas_backtest.items():
                    if isinstance(v, (np.integer, np.floating)):
                        metricas_json[k] = float(v)
                    else:
                        metricas_json[k] = v
                
                json.dump(metricas_json, f, indent=2)
        
        logger.info(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# NOTIFICACIONES
# ============================================

def enviar_telegram(mensaje: str, config: TradingConfig) -> bool:
    """Envía mensaje por Telegram"""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        logger.warning("  ⚠️ Telegram no configurado (TELEGRAM_TOKEN y TELEGRAM_CHAT_ID)")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": mensaje,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            logger.info("  📨 Telegram enviado")
            return True
        else:
            logger.warning(f"  ⚠️ Error Telegram: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ Error enviando Telegram: {e}")
        return False


# ============================================
# MAIN
# ============================================

def main():
    """Sistema principal mejorado"""
    logger.info("🚀 SISTEMA DE TRADING ALGORÍTMICO v2.0")
    logger.info("=" * 80)
    logger.info("✅ Validación estadística robusta")
    logger.info("✅ Costos de transacción realistas")
    logger.info("✅ Logging estructurado")
    logger.info("✅ Base de datos SQLite")
    logger.info("=" * 80)
    
    # Inicializar configuración
    config = TradingConfig()
    config.validate()
    
    # Inicializar DB
    db = TradingDatabase(config.DB_PATH)
    
    # Mostrar fechas
    fechas = config.get_fechas()
    logger.info(f"\n📅 Configuración:")
    logger.info(f"  Fecha actual: {fechas['actual'].strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"  Inicio entrenamiento: {fechas['inicio_entrenamiento'].strftime('%Y-%m-%d')}")
    logger.info(f"  Inicio validación: {fechas['inicio_validacion'].strftime('%Y-%m-%d')}")
    logger.info(f"  Inicio backtest: {fechas['inicio_backtest'].strftime('%Y-%m-%d')}")
    logger.info(f"  Días totales: {(fechas['actual'] - fechas['inicio_entrenamiento']).days}")
    
    # Crear directorio para modelos
    config.MODELOS_DIR.mkdir(exist_ok=True)
    
    resultados_globales = {}
    
    # Procesar cada activo
    for ticker in config.ACTIVOS:
        try:
            sistema = SistemaTradingTicker(ticker, config, db)
            
            # 1. Descargar datos
            if not sistema.descargar_datos():
                logger.warning(f"  ⏭️ Saltando {ticker}...")
                continue
            
            # 2. Entrenar modelos
            if not sistema.entrenar_modelos():
                logger.warning(f"  ⚠️ No se entrenaron modelos para {ticker}")
                continue
            
            # 3. Backtest
            if not sistema.ejecutar_backtest():
                logger.warning(f"  ⚠️ Backtest falló para {ticker}")
                continue
            
            # 4. Evaluar viabilidad
            viable, criterios = sistema.es_viable()
            
            logger.info(f"\n  {'✅ SISTEMA VIABLE' if viable else '❌ SISTEMA NO VIABLE'}")
            logger.info(f"  Criterios cumplidos: {criterios}/7")
            
            # 5. Guardar si es viable
            if viable:
                sistema.guardar_modelos()
            
            # Guardar resultados
            resultados_globales[ticker] = {
                'viable': viable,
                'criterios': criterios,
                'metricas': sistema.metricas_backtest
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando {ticker}: {e}", exc_info=True)
            continue
    
    # Resumen final
    logger.info(f"\n{'='*80}")
    logger.info("📊 RESUMEN FINAL")
    logger.info(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    
    logger.info(f"\n  Activos procesados: {len(resultados_globales)}")
    logger.info(f"  Sistemas viables: {len(viables)}")
    
    if viables:
        logger.info(f"\n  ✅ SISTEMAS VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            logger.info(f"\n    {ticker}:")
            logger.info(f"      Operaciones: {m['n_operaciones']}")
            logger.info(f"      Win rate: {m['tasa_exito']:.1%}")
            logger.info(f"      Retorno neto: {m['retorno_total_neto']:.2%}")
            logger.info(f"      Sharpe: {m['sharpe_ratio']:.2f}")
            logger.info(f"      Profit Factor: {m['profit_factor']:.2f}")
            logger.info(f"      p-value: {m['p_value']:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Proceso completado")
    logger.info(f"{'='*80}\n")
    
    return resultados_globales


if __name__ == "__main__":
    try:
        resultados = main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Proceso interrumpido por usuario")
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {e}", exc_info=True)
