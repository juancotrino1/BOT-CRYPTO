import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import requests
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
from pathlib import Path
import json
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def cargar_ultima_senal():
    if os.path.exists("ultima_senal.json"):
        with open("ultima_senal.json") as f:
            return json.load(f)
    return None

def guardar_ultima_senal(senal):
    with open("ultima_senal.json", "w") as f:
        json.dump(senal, f)

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("DEBUG token:", "OK" if token else "NONE")
    print("DEBUG chat_id:", chat_id)

    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": mensaje})

    print("📨 Telegram status:", r.status_code)
    print("📨 Telegram response:", r.text)


# ============================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================

class TradingConfig:
    """Configuración centralizada del sistema"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos de tiempo
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 365  # 1 año de datos históricos
    DIAS_VALIDACION = 90      # 3 meses para validación
    DIAS_BACKTEST = 90        # 3 meses para backtesting (aumentado de 30)
    
    # Activos
    ACTIVOS = [
        "BTC-USD"
    ]
    
    # Parámetros técnicos optimizados
    VENTANA_VOLATILIDAD = 20
    VENTANA_TENDENCIA = 48
    VENTANA_RAPIDA = 12
    ATR_PERIODO = 14
    RSI_PERIODO = 10  # Reducido para mayor sensibilidad
    
    # Horizontes de predicción optimizados
    HORIZONTES = [6, 12, 18, 36]  # Horas (optimizado para BTC)
    
    # Gestión de riesgo optimizada
    MULTIPLICADOR_SL = 1.8  # Reducido de 2.0
    MULTIPLICADOR_TP = 2.5  # Reducido de 3.0
    RATIO_MINIMO_RR = 1.5
    MAX_RIESGO_POR_OPERACION = 0.02  # 2% del capital
    
    # Validación
    N_FOLDS_WF = 3
    MIN_MUESTRAS_ENTRENAMIENTO = 500
    MIN_MUESTRAS_CLASE = 20
    
    # Umbrales optimizados
    UMBRAL_PROBABILIDAD_MIN = 0.62  # Bajado de 0.65
    UMBRAL_CONFIANZA_MIN = 0.58     # Bajado de 0.60
    
    # Filtros adicionales
    MIN_VOLUMEN_RELATIVO = 0.8      # 80% del volumen promedio
    MAX_RSI_EXTREMO = 75            # Evitar RSI > 75
    MIN_RSI_EXTREMO = 25            # Evitar RSI < 25
    
    # Persistencia
    MODELOS_DIR = Path("modelos_trading")
    
    @classmethod
    def get_fechas(cls):
        """Calcula fechas del sistema"""
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio_entrenamiento': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_validacion': now - timedelta(days=cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_backtest': now - timedelta(days=cls.DIAS_BACKTEST)
        }


# ============================================
# CÁLCULO DE INDICADORES MEJORADO
# ============================================

class IndicadoresTecnicos:
    """Calcula indicadores técnicos con nuevas features"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        """RSI robusto"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo, min_periods=periodo//2).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo, min_periods=periodo//2).mean()
        
        # Evitar división por cero
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
        """Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        close_prev = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=periodo, min_periods=periodo//2).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calcular_bollinger_bands(precios, ventana=20, num_std=2):
        """Bandas de Bollinger"""
        sma = precios.rolling(window=ventana, min_periods=ventana//2).mean()
        std = precios.rolling(window=ventana, min_periods=ventana//2).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        # Posición relativa en las bandas (0 = banda inferior, 1 = banda superior)
        bb_position = (precios - lower) / (upper - lower)
        bb_position = bb_position.clip(0, 1).fillna(0.5)
        
        return upper, lower, bb_position
    
    @staticmethod
    def calcular_features(df):
        """Calcula todas las features con nuevas mejoras"""
        df = df.copy()
        
        # Asegurar columnas simples
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(1, index=df.index))
        open_price = df['Open']
        
        # 1. Retornos
        df['retorno_1h'] = close.pct_change(1)
        df['retorno_4h'] = close.pct_change(4)
        df['retorno_24h'] = close.pct_change(24)
        
        # 2. Volatilidad
        df['volatilidad_24h'] = df['retorno_1h'].rolling(24, min_periods=12).std()
        df['volatilidad_normalizada'] = df['volatilidad_24h'] / df['volatilidad_24h'].rolling(100, min_periods=50).mean()
        
        # 3. Indicadores técnicos base
        df['RSI'] = IndicadoresTecnicos.calcular_rsi(close, TradingConfig.RSI_PERIODO)
        df['ATR'] = IndicadoresTecnicos.calcular_atr(df, TradingConfig.ATR_PERIODO)
        df['ATR_pct'] = df['ATR'] / close
        
        # 4. Medias móviles y tendencia
        df['SMA_12'] = close.rolling(12, min_periods=6).mean()
        df['SMA_48'] = close.rolling(48, min_periods=24).mean()
        df['EMA_12'] = close.ewm(span=12, min_periods=6).mean()
        
        df['dist_sma_12'] = (close - df['SMA_12']) / df['SMA_12']
        df['dist_sma_48'] = (close - df['SMA_48']) / df['SMA_48']
        df['tendencia'] = (df['SMA_12'] > df['SMA_48']).astype(int)
        
        # 5. Bollinger Bands
        bb_upper, bb_lower, bb_pos = IndicadoresTecnicos.calcular_bollinger_bands(close)
        df['BB_position'] = bb_pos
        df['BB_width'] = (bb_upper - bb_lower) / close
        
        # 6. Momentum
        df['momentum_6h'] = close / close.shift(6) - 1
        df['momentum_aceleracion'] = df['retorno_1h'].diff()
        
        # 7. Volumen
        df['volumen_relativo'] = volume / volume.rolling(24, min_periods=12).mean()
        
        # 8. Rango de precio
        df['rango_hl'] = (high - low) / close
        df['body_size'] = abs(close - open_price) / close
        
        # 9. Features de contexto
        df['hora_dia'] = df.index.hour
        df['es_apertura_ny'] = ((df['hora_dia'] >= 13) & (df['hora_dia'] <= 15)).astype(int)
        
        # 10. Z-scores para detección de anomalías
        for col in ['retorno_1h', 'volatilidad_24h', 'volumen_relativo']:
            if col in df.columns:
                media = df[col].rolling(100, min_periods=50).mean()
                std = df[col].rolling(100, min_periods=50).std()
                df[f'{col}_zscore'] = (df[col] - media) / (std + 1e-10)
        
        # 11. NUEVAS FEATURES CRÍTICAS
        
        # On-Balance Volume (OBV)
        df['obv'] = 0
        df.loc[close > close.shift(1), 'obv'] = volume
        df.loc[close < close.shift(1), 'obv'] = -volume
        df.loc[close == close.shift(1), 'obv'] = 0
        df['obv'] = df['obv'].cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)
        
        # Volume Price Trend (VPT)
        df['vpt'] = volume * ((close - close.shift(1)) / close.shift(1))
        df['vpt'] = df['vpt'].cumsum()
        
        # Accumulation/Distribution Line (ADL)
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
        df['adl'] = (clv * volume).cumsum()
        
        # Chaikin Money Flow (CMF)
        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.replace([np.inf, -np.inf], 0).fillna(0)
        mfv = mfv * volume
        df['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        # Vortex Indicator
        df['tr'] = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        df['vm_plus'] = (high - low.shift()).abs()
        df['vm_minus'] = (low - high.shift()).abs()
        
        df['vi_plus'] = df['vm_plus'].rolling(14).sum() / df['tr'].rolling(14).sum()
        df['vi_minus'] = df['vm_minus'].rolling(14).sum() / df['tr'].rolling(14).sum()
        df['vortex_signal'] = (df['vi_plus'] > df['vi_minus']).astype(int)
        
        # Detección de soportes y resistencias
        window = 20
        df['rolling_high'] = high.rolling(window).max()
        df['rolling_low'] = low.rolling(window).min()
        df['near_resistance'] = ((df['rolling_high'] - close) / close < 0.01).astype(int)
        df['near_support'] = ((close - df['rolling_low']) / close < 0.01).astype(int)
        
        # Market Structure
        df['higher_high'] = (high > high.rolling(5).max().shift(1)).astype(int)
        df['lower_low'] = (low < low.rolling(5).min().shift(1)).astype(int)
        df['market_structure'] = df['higher_high'] - df['lower_low']  # 1=alcista, -1=bajista, 0=neutral
        
        # Volume Profile Features
        df['volume_cluster'] = volume * df['body_size']
        df['high_volume_node'] = (df['volume_cluster'] > df['volume_cluster'].rolling(50).mean() * 1.5).astype(int)
        
        # Liquidity Features
        df['liquidity_score'] = volume / (df['volatilidad_24h'] + 1e-10)
        
        # Smart Money Concepts (simplificado)
        df['bosi'] = (close > open_price).astype(int)  # Bullish Order Block
        df['bearish_ob'] = (close < open_price).astype(int)  # Bearish Order Block
        
        # Session-based Features (para cripto 24/7)
        df['asian_session'] = ((df['hora_dia'] >= 0) & (df['hora_dia'] <= 8)).astype(int)
        df['europe_session'] = ((df['hora_dia'] >= 9) & (df['hora_dia'] <= 16)).astype(int)
        df['us_session'] = ((df['hora_dia'] >= 17) | (df['hora_dia'] <= 23)).astype(int)
        
        # Crypto-specific features
        df['trend_strength'] = abs(close.pct_change(24).rolling(24).mean()) / (df['volatilidad_24h'] + 1e-10)
        
        # Mean Reversion Features
        df["ret_log"] = np.log(close / close.shift(1))
        window_mr = 72
        df["mu"] = df["ret_log"].rolling(window_mr).mean()
        df["sigma"] = df["ret_log"].rolling(window_mr).std()
        df["sigma"] = df["sigma"].replace(0, np.nan)
        df["z_mr"] = (df["ret_log"] - df["mu"]) / df["sigma"]
        
        return df


# ============================================
# GESTIÓN DE RIESGO MEJORADA
# ============================================

class RiskManager:
    """Gestión de riesgo adaptativa"""
    
    @staticmethod
    def calcular_stop_loss_dinamico(precio, atr, tendencia, volatilidad, volatilidad_promedio):
        """Calcula SL basado en múltiples factores"""
        
        # Factores de ajuste
        factor_volatilidad = 1.0
        if volatilidad > volatilidad_promedio * 1.5:
            factor_volatilidad = 0.8  # Reducir SL en alta volatilidad
        
        factor_tendencia = 1.0
        if tendencia == 1:  # Alcista
            factor_tendencia = 1.2  # SL más amplio en tendencia
        
        # SL base
        sl_base = atr * TradingConfig.MULTIPLICADOR_SL
        
        # Ajustes
        sl_ajustado = sl_base * factor_volatilidad * factor_tendencia
        
        # Mínimo y máximo
        min_sl = precio * 0.002  # 0.2% mínimo
        max_sl = precio * 0.05   # 5% máximo
        
        return max(min_sl, min(sl_ajustado, max_sl))
    
    @staticmethod
    def evaluar_viabilidad_operacion(df_actual, señal, prob, confianza):
        """Evalúa si una operación es viable"""
        
        condiciones = []
        
        # 1. Volatilidad adecuada
        volatilidad_actual = df_actual['volatilidad_24h'].iloc[-1]
        volatilidad_promedio = df_actual['volatilidad_24h'].rolling(100).mean().iloc[-1]
        condiciones.append(0.5 < volatilidad_actual / volatilidad_promedio < 2.0)
        
        # 2. Volumen suficiente
        volumen_actual = df_actual['Volume'].iloc[-1]
        volumen_promedio = df_actual['Volume'].rolling(24).mean().iloc[-1]
        condiciones.append(volumen_actual > volumen_promedio * TradingConfig.MIN_VOLUMEN_RELATIVO)
        
        # 3. RSI no en extremos
        rsi_actual = df_actual['RSI'].iloc[-1]
        condiciones.append(TradingConfig.MIN_RSI_EXTREMO < rsi_actual < TradingConfig.MAX_RSI_EXTREMO)
        
        # 4. No en nivel clave (soporte/resistencia)
        cerca_resistencia = df_actual['near_resistance'].iloc[-1] if 'near_resistance' in df_actual.columns else 0
        cerca_soporte = df_actual['near_support'].iloc[-1] if 'near_support' in df_actual.columns else 0
        
        if señal == 'LONG':
            condiciones.append(cerca_resistencia == 0)
        else:  # SHORT
            condiciones.append(cerca_soporte == 0)
        
        # 5. Confianza suficiente
        condiciones.append(confianza > TradingConfig.UMBRAL_CONFIANZA_MIN)
        
        # 6. Probabilidad suficiente
        condiciones.append(prob > TradingConfig.UMBRAL_PROBABILIDAD_MIN)
        
        return all(condiciones)


# ============================================
# ETIQUETADO DE DATOS
# ============================================

class EtiquetadoDatos:
    """Crea etiquetas para entrenamiento"""
    
    @staticmethod
    def calcular_retorno_futuro(df, horizonte):
        """Calcula retorno futuro real"""
        return df['Close'].shift(-horizonte) / df['Close'] - 1
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte, umbral_movimiento=0.005):
        """
        Etiqueta binaria: 1 si hay movimiento significativo alcista, 0 si bajista
        Se ignoran movimientos pequeños (< umbral)
        """
        retorno_futuro = EtiquetadoDatos.calcular_retorno_futuro(df, horizonte)
        
        # Clasificación triple: LONG (1), SHORT (0), NEUTRAL (NaN)
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > umbral_movimiento] = 1
        etiqueta[retorno_futuro < -umbral_movimiento] = 0
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        """Prepara dataset completo para ML"""
        # Calcular features
        df = IndicadoresTecnicos.calcular_features(df)
        
        # Crear etiquetas
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Features para el modelo (sin look-ahead bias)
        features_base = [
            'RSI', 'ATR_pct', 'volatilidad_24h', 'volatilidad_normalizada',
            'dist_sma_12', 'dist_sma_48', 'tendencia',
            'BB_position', 'BB_width',
            'momentum_6h', 'momentum_aceleracion',
            'volumen_relativo', 'rango_hl', 'body_size',
            'retorno_1h', 'retorno_4h', 'retorno_24h',
            'retorno_1h_zscore', 'volatilidad_24h_zscore', 'volumen_relativo_zscore',
            'es_apertura_ny', 'obv_trend', 'cmf',
            'vortex_signal', 'near_resistance', 'near_support',
            'market_structure', 'high_volume_node', 'liquidity_score',
            'bosi', 'bearish_ob', 'asian_session', 'europe_session', 'us_session',
            'trend_strength', 'z_mr'
        ]
        
        # Filtrar solo features disponibles
        features_disponibles = [f for f in features_base if f in df.columns]
        
        return df, features_disponibles


# ============================================
# MODELO DE MACHINE LEARNING MEJORADO
# ============================================

class ModeloPrediccion:
    """Modelo de ML mejorado con múltiples algoritmos"""
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
        self.modelo_nombre = ""
    
    def entrenar_walk_forward(self, df, features, etiqueta_col):
        """Entrenamiento con validación walk-forward y selección de mejor modelo"""
        
        # Filtrar datos válidos
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)} < {TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO}")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        if y.sum() < TradingConfig.MIN_MUESTRAS_CLASE or (len(y) - y.sum()) < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas: Positivos={y.sum()}, Negativos={len(y)-y.sum()}")
            return False
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=TradingConfig.N_FOLDS_WF)
        
        # Modelos a probar
        modelos_config = [
            {
                'nombre': 'XGBoost',
                'modelo': XGBClassifier(
                    n_estimators=200,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            },
            {
                'nombre': 'GradientBoosting',
                'modelo': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
            },
            {
                'nombre': 'RandomForest',
                'modelo': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            }
        ]
        
        mejores_resultados = {'nombre': '', 'accuracy': 0, 'modelo': None, 'scores': []}
        
        for config in modelos_config:
            print(f"    Probando {config['nombre']}...")
            scores_modelo = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Escalar
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Entrenar modelo
                modelo = config['modelo']
                modelo.fit(X_train_scaled, y_train)
                
                # Evaluar
                y_pred = modelo.predict(X_val_scaled)
                acc = accuracy_score(y_val, y_pred)
                scores_modelo.append(acc)
            
            acc_promedio = np.mean(scores_modelo)
            print(f"      {config['nombre']}: Accuracy={acc_promedio:.2%}")
            
            if acc_promedio > mejores_resultados['accuracy']:
                mejores_resultados = {
                    'nombre': config['nombre'],
                    'accuracy': acc_promedio,
                    'modelo': config['modelo'],
                    'scores': scores_modelo
                }
        
        print(f"    ✅ Mejor modelo: {mejores_resultados['nombre']} ({mejores_resultados['accuracy']:.2%})")
        
        # Entrenar modelo final con todos los datos
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = mejores_resultados['modelo']
        self.modelo.fit(X_scaled, y)
        self.features = features
        self.modelo_nombre = mejores_resultados['nombre']
        
        # Guardar métricas
        self.metricas_validacion = {
            'modelo': mejores_resultados['nombre'],
            'accuracy': mejores_resultados['accuracy'],
            'accuracy_scores': mejores_resultados['scores']
        }
        
        return True
    
    def predecir(self, df_actual):
        """Realiza predicción en datos nuevos"""
        if self.modelo is None:
            return None
        
        # Asegurar que tenemos todas las features
        if not all(f in df_actual.columns for f in self.features):
            return None
        
        X = df_actual[self.features].iloc[[-1]]  # Última fila
        X_scaled = self.scaler.transform(X)
        
        prediccion_clase = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'prediccion': int(prediccion_clase),
            'probabilidad_positiva': probabilidades[1],
            'probabilidad_negativa': probabilidades[0],
            'confianza': max(probabilidades),
            'modelo': self.modelo_nombre
        }
    
    def guardar(self, path):
        """Guarda el modelo"""
        if self.modelo is None:
            return False
        
        modelo_data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features': self.features,
            'metricas': self.metricas_validacion,
            'horizonte': self.horizonte,
            'ticker': self.ticker,
            'modelo_nombre': self.modelo_nombre
        }
        
        joblib.dump(modelo_data, path)
        return True
    
    @classmethod
    def cargar(cls, path):
        """Carga un modelo guardado"""
        modelo_data = joblib.load(path)
        
        instancia = cls(modelo_data['horizonte'], modelo_data['ticker'])
        instancia.modelo = modelo_data['modelo']
        instancia.scaler = modelo_data['scaler']
        instancia.features = modelo_data['features']
        instancia.metricas_validacion = modelo_data['metricas']
        instancia.modelo_nombre = modelo_data.get('modelo_nombre', 'Desconocido')
        
        return instancia


# ============================================
# BACKTESTING MEJORADO
# ============================================

class Backtester:
    """Ejecuta backtesting con métricas mejoradas"""
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos  # Dict de modelos por horizonte
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, features_row):
        """Simula una operación completa con gestión de riesgo mejorada"""
        precio_entrada = self.df.loc[idx, 'Close']
        atr = self.df.loc[idx, 'ATR']
        
        # Determinar dirección
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Calcular niveles con risk manager
        volatilidad_actual = features_row['volatilidad_24h']
        volatilidad_promedio = self.df['volatilidad_24h'].rolling(100).mean().iloc[-1]
        tendencia = features_row.get('tendencia', 0)
        
        # Calcular stop loss dinámico
        sl_distance = RiskManager.calcular_stop_loss_dinamico(
            precio_entrada, atr, tendencia, 
            volatilidad_actual, volatilidad_promedio
        )
        
        if señal_long:
            stop_loss = precio_entrada - sl_distance
            take_profit = precio_entrada + (sl_distance * (TradingConfig.MULTIPLICADOR_TP / TradingConfig.MULTIPLICADOR_SL))
        else:
            stop_loss = precio_entrada + sl_distance
            take_profit = precio_entrada - (sl_distance * (TradingConfig.MULTIPLICADOR_TP / TradingConfig.MULTIPLICADOR_SL))
        
        riesgo = abs(precio_entrada - stop_loss)
        recompensa = abs(take_profit - precio_entrada)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        # Filtro R:R
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Validar distancia mínima
        min_dist = precio_entrada * 0.001
        if abs(take_profit - precio_entrada) < min_dist or abs(stop_loss - precio_entrada) < min_dist:
            return None
        
        # Simular resultado (mirar hacia adelante máximo 48 horas)
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(48, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Close'].values
        
        # Determinar resultado
        resultado = 'TIEMPO'
        velas_hasta_cierre = max_ventana
        retorno = 0
        
        for i, precio in enumerate(precios_futuros[1:], 1):
            if señal_long:
                if precio >= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = (take_profit - precio_entrada) / precio_entrada
                    break
                elif precio <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = (stop_loss - precio_entrada) / precio_entrada
                    break
            else:  # SHORT
                if precio <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = (precio_entrada - take_profit) / precio_entrada
                    break
                elif precio >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = (precio_entrada - stop_loss) / precio_entrada
                    break
        
        # Si llegamos hasta el final sin hit
        if resultado == 'TIEMPO':
            precio_cierre = precios_futuros[velas_hasta_cierre]
            if señal_long:
                retorno = (precio_cierre - precio_entrada) / precio_entrada
            else:
                retorno = (precio_entrada - precio_cierre) / precio_entrada
        
        return {
            'fecha': idx,
            'ticker': self.ticker,
            'direccion': direccion,
            'precio_entrada': precio_entrada,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ratio_rr': ratio_rr,
            'probabilidad': prob,
            'resultado': resultado,
            'retorno': retorno,
            'velas_hasta_cierre': velas_hasta_cierre,
            'volatilidad': volatilidad_actual,
            'volumen': features_row['Volume'],
            'rsi': features_row.get('RSI', 50)
        }
    
    def ejecutar(self, fecha_inicio, umbral_prob=0.62):
        """Ejecuta backtesting completo"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtesting: {len(df_backtest)} velas")
            return None
        
        print(f"  📊 Backtesting: {df_backtest.index[0]} a {df_backtest.index[-1]} ({len(df_backtest)} velas)")
        
        # Iterar sobre cada vela
        for idx in df_backtest.index[:-48]:  # Dejar margen para simulación
            predicciones = {}
            
            # Obtener predicciones de todos los horizontes
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Consenso de modelos
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # Decidir señal
            señal_long = prob_promedio > 0.5
            prob_real = prob_promedio if señal_long else 1 - prob_promedio
            
            # Filtros de viabilidad
            if not RiskManager.evaluar_viabilidad_operacion(
                df_backtest.loc[:idx], 
                'LONG' if señal_long else 'SHORT',
                prob_real,
                confianza_promedio
            ):
                continue
            
            # Simular operación
            operacion = self.simular_operacion(
                idx, 
                señal_long, 
                prob_real,
                df_backtest.loc[idx]
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones en backtesting")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
        """Calcula métricas de rendimiento mejoradas"""
        df_ops = pd.DataFrame(self.operaciones)
        
        if df_ops.empty:
            return None
        
        n_ops = len(df_ops)
        n_tp = (df_ops['resultado'] == 'TP').sum()
        n_sl = (df_ops['resultado'] == 'SL').sum()
        
        retornos = df_ops['retorno']
        operaciones_ganadoras = retornos > 0
        
        # Métricas básicas
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': operaciones_ganadoras.sum() / n_ops,
            'hit_tp_rate': n_tp / n_ops,
            'hit_sl_rate': n_sl / n_ops,
            'retorno_total': retornos.sum(),
            'retorno_promedio': retornos.mean(),
            'retorno_mediano': retornos.median(),
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'profit_factor': abs(retornos[retornos > 0].sum() / retornos[retornos < 0].sum()) if (retornos < 0).any() else np.inf,
            'max_drawdown': self._calcular_max_drawdown(retornos),
            'sharpe_ratio': retornos.mean() / retornos.std() if retornos.std() > 0 else 0,
        }
        
        # NUEVAS MÉTRICAS AVANZADAS
        # 1. Expectativa matemática
        df_ops['expectativa'] = df_ops['retorno'] * df_ops['probabilidad']
        metricas['expectativa_promedio'] = df_ops['expectativa'].mean()
        
        # 2. Ratio de recuperación
        ganancias = df_ops[df_ops['retorno'] > 0]['retorno']
        perdidas = df_ops[df_ops['retorno'] < 0]['retorno']
        metricas['recovery_factor'] = abs(ganancias.sum() / perdidas.sum()) if perdidas.sum() != 0 else np.inf
        
        # 3. Ratio de Calmar
        metricas['calmar_ratio'] = metricas['retorno_total'] / abs(metricas['max_drawdown']) if metricas['max_drawdown'] != 0 else np.inf
        
        # 4. Consistencia
        metricas['consistencia'] = 1 / retornos.std() if retornos.std() > 0 else 0
        
        # 5. Análisis por volatilidad
        df_ops['regime'] = pd.qcut(df_ops['volatilidad'], q=3, labels=['bajo', 'medio', 'alto'])
        metricas['win_rate_alta_vol'] = df_ops[df_ops['regime'] == 'alto']['retorno'].gt(0).mean()
        metricas['win_rate_baja_vol'] = df_ops[df_ops['regime'] == 'bajo']['retorno'].gt(0).mean()
        
        # 6. Ratio de operaciones óptimas
        condiciones_optimas = (
            (df_ops['probabilidad'] > 0.6) &
            (df_ops['ratio_rr'] > 1.5) &
            (df_ops['velas_hasta_cierre'] < 24)
        )
        metricas['operaciones_optimas_pct'] = condiciones_optimas.mean() if len(df_ops) > 0 else 0
        
        # 7. Eficiencia por dirección
        metricas['win_rate_long'] = df_ops[df_ops['direccion'] == 'LONG']['retorno'].gt(0).mean()
        metricas['win_rate_short'] = df_ops[df_ops['direccion'] == 'SHORT']['retorno'].gt(0).mean()
        
        # 8. Slippage estimado
        metricas['slippage_promedio'] = df_ops['volatilidad'].mean() * 0.1
        
        return metricas, df_ops
    
    def _calcular_max_drawdown(self, retornos):
        """Calcula drawdown máximo"""
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()


# ============================================
# SISTEMA COMPLETO POR TICKER
# ============================================

class SistemaTradingTicker:
    """Sistema completo de trading para un ticker"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.modelos = {}
        self.fechas = TradingConfig.get_fechas()
        self.df_historico = None
        self.metricas_backtest = None
    
    def descargar_datos(self):
        """Descarga datos históricos"""
        print(f"\n{'='*80}")
        print(f"📥 DESCARGANDO {self.ticker}")
        print(f"{'='*80}")
        
        try:
            df = yf.download(
                self.ticker,
                start=self.fechas['inicio_entrenamiento'],
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df.empty:
                print(f"  ❌ No hay datos disponibles")
                return False
            
            # Limpiar MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas")
            print(f"  📅 Rango: {df.index[0]} a {df.index[-1]}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def entrenar_modelos(self):
        """Entrena modelos para todos los horizontes"""
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Usar datos hasta inicio de backtest para entrenamiento
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        
        modelos_entrenados = 0
        
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            # Preparar dataset
            df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
            etiqueta_col = f'etiqueta_{horizonte}h'
            
            # Entrenar modelo
            modelo = ModeloPrediccion(horizonte, self.ticker)
            if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                self.modelos[horizonte] = modelo
                modelos_entrenados += 1
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/{len(TradingConfig.HORIZONTES)}")
        
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        """Ejecuta backtesting"""
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            print("  ❌ No hay modelos entrenados")
            return False
        
        # Preparar datos completos (incluye período de backtest)
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(
            self.df_historico, 
            TradingConfig.HORIZONTES[0]
        )
        
        # Ejecutar backtest
        backtester = Backtester(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS AVANZADOS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Tasa éxito: {metricas['tasa_exito']:.2%}")
        print(f"    Hit TP: {metricas['hit_tp_rate']:.2%}")
        print(f"    Hit SL: {metricas['hit_sl_rate']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        print(f"    Calmar Ratio: {metricas['calmar_ratio']:.2f}")
        print(f"    Win Rate Long: {metricas.get('win_rate_long', 0):.2%}")
        print(f"    Win Rate Short: {metricas.get('win_rate_short', 0):.2%}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable para trading"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        criterios = []
        
        # Criterio 1: Tasa de éxito > 50%
        criterios.append(m['tasa_exito'] > 0.50)
        
        # Criterio 2: Retorno total positivo
        criterios.append(m['retorno_total'] > 0)
        
        # Criterio 3: Profit factor > 1.2
        criterios.append(m['profit_factor'] > 1.2)
        
        # Criterio 4: Drawdown controlado
        criterios.append(abs(m['max_drawdown']) < 0.25)
        
        # Criterio 5: Suficientes operaciones
        criterios.append(m['n_operaciones'] >= 20)
        
        # Criterio 6: Sharpe ratio positivo
        criterios.append(m['sharpe_ratio'] > 0)
        
        # Criterio 7: Calmar ratio > 0.5
        criterios.append(m['calmar_ratio'] > 0.5)
        
        # Criterio 8: Operaciones óptimas > 30%
        criterios.append(m.get('operaciones_optimas_pct', 0) > 0.3)
        
        criterios_cumplidos = sum(criterios)
        viable = criterios_cumplidos >= 5  # Al menos 5 de 8 criterios
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        """Análisis en tiempo real para señal actual"""
        if not self.modelos:
            return None
        
        try:
            df_reciente = yf.download(
                self.ticker,
                start=self.fechas['actual'] - timedelta(days=7),
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df_reciente.empty:
                return None
            
            if isinstance(df_reciente.columns, pd.MultiIndex):
                df_reciente.columns = df_reciente.columns.get_level_values(0)
            
            df_reciente = df_reciente[['Open', 'High', 'Low', 'Close', 'Volume']]
            df_reciente = IndicadoresTecnicos.calcular_features(df_reciente)
            
            # === PREDICCIONES ===
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_reciente)
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                return None
            
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            señal = "LONG" if prob_promedio > 0.5 else "SHORT"
            prob_real = prob_promedio if señal == "LONG" else 1 - prob_promedio
            
            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            atr = ultima_vela['ATR']
            
            if pd.isna(atr) or atr <= 0:
                return None
            
            # Calcular SL y TP con risk manager
            volatilidad_actual = ultima_vela['volatilidad_24h']
            volatilidad_promedio = df_reciente['volatilidad_24h'].rolling(100).mean().iloc[-1]
            tendencia = ultima_vela.get('tendencia', 0)
            
            # Stop loss dinámico
            sl_distance = RiskManager.calcular_stop_loss_dinamico(
                precio, atr, tendencia, 
                volatilidad_actual, volatilidad_promedio
            )
            
            if señal == 'LONG':
                sl = precio - sl_distance
                tp = precio + (sl_distance * (TradingConfig.MULTIPLICADOR_TP / TradingConfig.MULTIPLICADOR_SL))
            else:
                sl = precio + sl_distance
                tp = precio - (sl_distance * (TradingConfig.MULTIPLICADOR_TP / TradingConfig.MULTIPLICADOR_SL))
            
            # Validar distancia mínima
            min_dist = precio * 0.001
            if abs(tp - precio) < min_dist or abs(sl - precio) < min_dist:
                return None
            
            ratio_rr = abs(tp - precio) / abs(precio - sl)
            
            # Evaluar viabilidad
            if not RiskManager.evaluar_viabilidad_operacion(
                df_reciente,
                señal,
                prob_real,
                confianza_promedio
            ):
                return None
            
            # Verificar ratio R:R mínimo
            if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
                return None
            
            return {
                'ticker': self.ticker,
                'fecha': datetime.now(TradingConfig.TIMEZONE),
                'precio': precio,
                'señal': señal,
                'probabilidad': prob_real,
                'confianza': confianza_promedio,
                'stop_loss': sl,
                'take_profit': tp,
                'ratio_rr': ratio_rr,
                'predicciones_detalle': predicciones,
                'rsi': ultima_vela.get('RSI', 50),
                'tendencia': 'ALCISTA' if ultima_vela.get('tendencia', 0) == 1 else 'BAJISTA',
                'volatilidad': volatilidad_actual,
                'volumen_relativo': ultima_vela.get('volumen_relativo', 1),
                'z_mr': ultima_vela.get('z_mr', 0),
                'evento_mr': "MR SHORT" if ultima_vela.get('z_mr', 0) > 2.2 else "MR LONG" if ultima_vela.get('z_mr', 0) < -2.2 else "NO",
                'modelos_usados': [f"{h}h:{p['modelo']}" for h, p in predicciones.items()]
            }
            
        except Exception as e:
            print(f"  ❌ Error análisis tiempo real: {e}")
            return None
    
    def guardar_modelos(self):
        """Guarda modelos entrenados"""
        if not self.modelos:
            return False
        
        path_ticker = TradingConfig.MODELOS_DIR / self.ticker
        path_ticker.mkdir(parents=True, exist_ok=True)
        
        for horizonte, modelo in self.modelos.items():
            path_modelo = path_ticker / f"modelo_{horizonte}h.pkl"
            modelo.guardar(path_modelo)
        
        print(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    print("🚀 SISTEMA DE TRADING OPTIMIZADO")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración temporal:")
    print(f"  Actual: {fechas['actual'].date()}")
    print(f"  Entrenamiento desde: {fechas['inicio_entrenamiento'].date()}")
    print(f"  Backtest desde: {fechas['inicio_backtest'].date()}")
    print(f"  Intervalo: {TradingConfig.INTERVALO}")
    print(f"  Horizontes: {TradingConfig.HORIZONTES} horas")
    print(f"  Modelos a probar: XGBoost, GradientBoosting, RandomForest")
    
    resultados_globales = {}
    
    # Procesar cada ticker
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        # 1. Descargar datos
        if not sistema.descargar_datos():
            continue
        
        # 2. Entrenar modelos
        if not sistema.entrenar_modelos():
            print(f"  ❌ No se pudieron entrenar modelos para {ticker}")
            continue
        
        # 3. Backtest
        if not sistema.ejecutar_backtest():
            print(f"  ❌ Backtest fallido para {ticker}")
            continue
        
        # 4. Evaluar viabilidad
        viable, criterios = sistema.es_viable()
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUACIÓN - {ticker}")
        print(f"{'='*80}")
        print(f"  Criterios cumplidos: {criterios}/8")
        print(f"  Viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        # 5. Análisis tiempo real (solo si es viable)
        señal_actual = None
        
        if viable:
            señal_actual = sistema.analizar_tiempo_real()
            
            if señal_actual:
                print(f"\n  🚨 SEÑAL DETECTADA:")
                print(f"    Dirección: {señal_actual['señal']}")
                print(f"    Probabilidad: {señal_actual['probabilidad']:.2%}")
                print(f"    Confianza: {señal_actual['confianza']:.2%}")
                print(f"    Precio: ${señal_actual['precio']:,.2f}")
                print(f"    SL: ${señal_actual['stop_loss']:,.2f}")
                print(f"    TP: ${señal_actual['take_profit']:,.2f}")
                print(f"    R:R: {señal_actual['ratio_rr']:.2f}")
                print(f"    Modelos: {', '.join(señal_actual['modelos_usados'])}")
                
                # 🔁 Control de repetición
                ultima = cargar_ultima_senal()
                if ultima and ultima["ticker"] == ticker and ultima["señal"] == señal_actual["señal"]:
                    print("🔁 Señal repetida. No se envía.")
                else:
                    fecha = señal_actual['fecha'].strftime("%Y-%m-%d %H:%M")
                    
                    mensaje = (
                        f"📊 SEÑAL {ticker}\n"
                        f"🕒 Fecha: {fecha}\n"
                        f"⏱ TF: {TradingConfig.INTERVALO}\n"
                        f"📈 Tendencia: {señal_actual['tendencia']}\n"
                        f"📊 RSI: {señal_actual['rsi']:.1f}\n"
                        f"📈 Volatilidad: {señal_actual['volatilidad']:.2%}\n"
                        f"📊 Volumen relativo: {señal_actual['volumen_relativo']:.1f}x\n\n"
                        f"Dirección: {señal_actual['señal']}\n"
                        f"Probabilidad: {señal_actual['probabilidad']:.2%}\n"
                        f"Confianza: {señal_actual['confianza']:.2%}\n\n"
                        f"🎯 Entrada: {señal_actual['precio']:.2f}\n"
                        f"🛑 SL: {señal_actual['stop_loss']:.2f}\n"
                        f"🎯 TP: {señal_actual['take_profit']:.2f}\n"
                        f"⚖️ R:R: {señal_actual['ratio_rr']:.2f}\n"
                        f"📐 Mean Reversion: {señal_actual['evento_mr']}\n"
                        f"📐 Z-score: {señal_actual['z_mr']:.2f}\n"
                        f"🤖 Modelos: {', '.join(señal_actual['modelos_usados'])}\n\n"
                    )
                    
                    enviar_telegram(mensaje)
                    
                    guardar_ultima_senal({
                        "ticker": ticker,
                        "señal": señal_actual["señal"],
                        "fecha": str(señal_actual["fecha"])
                    })
            else:
                print(f"\n  📭 No hay señal viable en este momento")
        
        # 6. Guardar modelos
        if viable:
            sistema.guardar_modelos()
        
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest,
            'señal_actual': señal_actual
        }
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN GLOBAL")
    print(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    
    print(f"\n  Tickers procesados: {len(resultados_globales)}")
    print(f"  Tickers viables: {len(viables)}")
    
    if viables:
        print(f"\n  ✅ TICKERS VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            print(f"    {ticker}: Retorno {m['retorno_total']:.2%}, "
                  f"Win rate {m['tasa_exito']:.2%}, "
                  f"PF {m['profit_factor']:.2f}, "
                  f"DD {m['max_drawdown']:.2%}")
    
    return resultados_globales


if __name__ == "__main__":
    main()
