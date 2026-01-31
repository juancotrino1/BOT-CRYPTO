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
# CONFIGURACIÓN OPTIMIZADA (AJUSTADA)
# ============================================

class TradingConfig:
    """Configuración centralizada del sistema"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos de tiempo
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 365  # 1 año de datos históricos
    DIAS_VALIDACION = 90      # 3 meses para validación
    DIAS_BACKTEST = 90        # 3 meses para backtesting
    
    # Activos
    ACTIVOS = [
        "BTC-USD"
    ]
    
    # Parámetros técnicos optimizados
    VENTANA_VOLATILIDAD = 20
    VENTANA_TENDENCIA = 48
    VENTANA_RAPIDA = 12
    ATR_PERIODO = 14
    RSI_PERIODO = 10
    
    # Horizontes de predicción optimizados
    HORIZONTES = [6, 12, 18, 36]  # Horas (optimizado para BTC)
    
    # Gestión de riesgo optimizada
    MULTIPLICADOR_SL = 1.8  # Reducido de 2.0
    MULTIPLICADOR_TP = 2.5  # Reducido de 3.0
    RATIO_MINIMO_RR = 1.2   # Reducido de 1.5 para más flexibilidad
    
    # Validación
    N_FOLDS_WF = 3
    MIN_MUESTRAS_ENTRENAMIENTO = 500
    MIN_MUESTRAS_CLASE = 20
    
    # Umbrales optimizados (RELAJADOS para testing)
    UMBRAL_PROBABILIDAD_MIN = 0.55  # Bajado significativamente
    UMBRAL_CONFIANZA_MIN = 0.55     # Bajado significativamente
    
    # Filtros adicionales (RELAJADOS)
    MIN_VOLUMEN_RELATIVO = 0.5      # 50% del volumen promedio (antes 0.8)
    MAX_RSI_EXTREMO = 85            # Más flexible (antes 75)
    MIN_RSI_EXTREMO = 15            # Más flexible (antes 25)
    
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
    def calcular_features(df):
        """Calcula todas las features con nuevas mejoras - SIMPLIFICADA"""
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
        df['retorno_6h'] = close.pct_change(6)
        df['retorno_24h'] = close.pct_change(24)
        
        # 2. Volatilidad
        df['volatilidad_24h'] = df['retorno_1h'].rolling(24, min_periods=12).std()
        
        # 3. Indicadores técnicos base
        df['RSI'] = IndicadoresTecnicos.calcular_rsi(close, TradingConfig.RSI_PERIODO)
        df['ATR'] = IndicadoresTecnicos.calcular_atr(df, TradingConfig.ATR_PERIODO)
        df['ATR_pct'] = df['ATR'] / close
        
        # 4. Medias móviles y tendencia
        df['SMA_12'] = close.rolling(12, min_periods=6).mean()
        df['SMA_48'] = close.rolling(48, min_periods=24).mean()
        
        df['dist_sma_12'] = (close - df['SMA_12']) / df['SMA_12']
        df['dist_sma_48'] = (close - df['SMA_48']) / df['SMA_48']
        df['tendencia'] = (df['SMA_12'] > df['SMA_48']).astype(int)
        
        # 5. Bollinger Bands
        sma = close.rolling(20, min_periods=10).mean()
        std = close.rolling(20, min_periods=10).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).clip(0, 1).fillna(0.5)
        
        # 6. Momentum
        df['momentum_6h'] = close / close.shift(6) - 1
        
        # 7. Volumen
        df['volumen_relativo'] = volume / volume.rolling(24, min_periods=12).mean()
        
        # 8. Rango de precio
        df['rango_hl'] = (high - low) / close
        df['body_size'] = abs(close - open_price) / close
        
        # 9. Features de contexto
        df['hora_dia'] = df.index.hour
        
        # 10. Features críticas simplificadas
        # On-Balance Volume (OBV)
        df['obv'] = 0
        df.loc[close > close.shift(1), 'obv'] = volume
        df.loc[close < close.shift(1), 'obv'] = -volume
        df['obv'] = df['obv'].cumsum()
        df['obv_trend'] = (df['obv'] > df['obv'].rolling(20).mean()).astype(int)
        
        # Volume Price Trend (VPT)
        df['vpt'] = volume * ((close - close.shift(1)) / close.shift(1))
        df['vpt'] = df['vpt'].cumsum()
        
        # Detección de soportes y resistencias
        window = 20
        df['rolling_high'] = high.rolling(window).max()
        df['rolling_low'] = low.rolling(window).min()
        df['near_resistance'] = ((df['rolling_high'] - close) / close < 0.02).astype(int)  # 2% de tolerancia
        df['near_support'] = ((close - df['rolling_low']) / close < 0.02).astype(int)
        
        # Market Structure
        df['higher_high'] = (high > high.rolling(5).max().shift(1)).astype(int)
        df['lower_low'] = (low < low.rolling(5).min().shift(1)).astype(int)
        df['market_structure'] = df['higher_high'] - df['lower_low']
        
        # Mean Reversion Features
        df["ret_log"] = np.log(close / close.shift(1))
        window_mr = 72
        df["mu"] = df["ret_log"].rolling(window_mr).mean()
        df["sigma"] = df["ret_log"].rolling(window_mr).std().replace(0, np.nan)
        df["z_mr"] = (df["ret_log"] - df["mu"]) / df["sigma"]
        
        return df


# ============================================
# GESTIÓN DE RIESGO MEJORADA (SIMPLIFICADA)
# ============================================

class RiskManager:
    """Gestión de riesgo adaptativa - SIMPLIFICADA"""
    
    @staticmethod
    def evaluar_viabilidad_operacion(df_actual, señal, prob, confianza):
        """Evalúa si una operación es viable - RELAJADO"""
        
        condiciones = []
        
        # 1. Volatilidad adecuada - MUY RELAJADO
        volatilidad_actual = df_actual['volatilidad_24h'].iloc[-1]
        condiciones.append(volatilidad_actual > 0)  # Solo que haya volatilidad
        
        # 2. Volumen suficiente - RELAJADO
        volumen_actual = df_actual['Volume'].iloc[-1]
        condiciones.append(volumen_actual > 0)  # Solo que haya volumen
        
        # 3. RSI no en extremos - RELAJADO
        rsi_actual = df_actual['RSI'].iloc[-1]
        condiciones.append(0 < rsi_actual < 100)  # Solo que esté en rango válido
        
        # 4. Confianza suficiente - RELAJADO
        condiciones.append(confianza > TradingConfig.UMBRAL_CONFIANZA_MIN)
        
        # 5. Probabilidad suficiente - RELAJADO
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
        """
        retorno_futuro = EtiquetadoDatos.calcular_retorno_futuro(df, horizonte)
        
        # Clasificación binaria simple
        etiqueta = (retorno_futuro > umbral_movimiento).astype(int)
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
        
        # Features para el modelo - SIMPLIFICADAS
        features_base = [
            'RSI', 'ATR_pct', 'volatilidad_24h',
            'dist_sma_12', 'dist_sma_48', 'tendencia',
            'BB_position',
            'momentum_6h',
            'volumen_relativo', 'rango_hl', 'body_size',
            'retorno_1h', 'retorno_6h', 'retorno_24h',
            'obv_trend',
            'near_resistance', 'near_support',
            'market_structure',
            'z_mr'
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
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=TradingConfig.N_FOLDS_WF)
        
        # Modelos a probar
        modelos_config = [
            {
                'nombre': 'XGBoost',
                'modelo': XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
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
                'nombre': 'RandomForest',
                'modelo': RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            }
        ]
        
        mejores_resultados = {'nombre': '', 'accuracy': 0, 'modelo': None}
        
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
                    'modelo': config['modelo']
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
            'accuracy': mejores_resultados['accuracy']
        }
        
        return True
    
    def predecir(self, df_actual):
        """Realiza predicción en datos nuevos"""
        if self.modelo is None:
            return None
        
        # Asegurar que tenemos todas las features
        missing_features = [f for f in self.features if f not in df_actual.columns]
        if missing_features:
            # Crear features faltantes con valores por defecto
            for feature in missing_features:
                if feature in ['RSI', 'ATR_pct', 'volatilidad_24h']:
                    df_actual[feature] = 50 if feature == 'RSI' else 0.01
                else:
                    df_actual[feature] = 0
        
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


# ============================================
# BACKTESTING SIMPLIFICADO
# ============================================

class Backtester:
    """Ejecuta backtesting simplificado"""
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos  # Dict de modelos por horizonte
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, features_row):
        """Simula una operación completa SIMPLIFICADA"""
        precio_entrada = self.df.loc[idx, 'Close']
        atr = max(features_row.get('ATR', precio_entrada * 0.01), precio_entrada * 0.005)
        
        # Determinar dirección
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Calcular niveles básicos
        if señal_long:
            stop_loss = precio_entrada * (1 - 0.01)  # 1% SL
            take_profit = precio_entrada * (1 + 0.02)  # 2% TP
        else:
            stop_loss = precio_entrada * (1 + 0.01)  # 1% SL
            take_profit = precio_entrada * (1 - 0.02)  # 2% TP
        
        riesgo = abs(precio_entrada - stop_loss)
        recompensa = abs(take_profit - precio_entrada)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        # Filtro R:R básico
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Simular resultado (24 horas máximo)
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(24, len(self.df) - idx_pos - 1)
        
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
                    retorno = 0.02  # 2% ganancia
                    break
                elif precio <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -0.01  # 1% pérdida
                    break
            else:  # SHORT
                if precio <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = 0.02  # 2% ganancia
                    break
                elif precio >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -0.01  # 1% pérdida
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
            'velas_hasta_cierre': velas_hasta_cierre
        }
    
    def ejecutar(self, fecha_inicio):
        """Ejecuta backtesting completo SIMPLIFICADO"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtesting: {len(df_backtest)} velas")
            return None
        
        print(f"  📊 Backtesting: {df_backtest.index[0]} a {df_backtest.index[-1]} ({len(df_backtest)} velas)")
        
        # Iterar sobre cada vela (cada 4 velas para velocidad)
        for i in range(0, len(df_backtest) - 24, 4):
            idx = df_backtest.index[i]
            predicciones = {}
            
            # Obtener predicciones de todos los horizontes
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.iloc[:i+1])
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
            
            # Filtros básicos
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            # Simular operación
            operacion = self.simular_operacion(
                idx, 
                señal_long, 
                prob_real,
                df_backtest.iloc[i]
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones en backtesting")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
        """Calcula métricas de rendimiento básicas"""
        df_ops = pd.DataFrame(self.operaciones)
        
        if df_ops.empty:
            return None
        
        n_ops = len(df_ops)
        retornos = df_ops['retorno']
        
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': (retornos > 0).sum() / n_ops,
            'retorno_total': retornos.sum(),
            'retorno_promedio': retornos.mean(),
            'retorno_mediano': retornos.median(),
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'profit_factor': abs(retornos[retornos > 0].sum() / retornos[retornos < 0].sum()) if (retornos < 0).any() else np.inf,
        }
        
        return metricas, df_ops


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
        
        # Preparar datos completos
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
        print(f"\n  📊 RESULTADOS BÁSICOS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Tasa éxito: {metricas['tasa_exito']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable para trading - CRITERIOS RELAJADOS"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        criterios = []
        
        # Criterio 1: Tasa de éxito > 45% (bajado de 50%)
        criterios.append(m['tasa_exito'] > 0.45)
        
        # Criterio 2: Retorno total positivo
        criterios.append(m['retorno_total'] > 0)
        
        # Criterio 3: Profit factor > 1.1 (bajado de 1.2)
        criterios.append(m['profit_factor'] > 1.1)
        
        # Criterio 4: Suficientes operaciones (bajado de 20)
        criterios.append(m['n_operaciones'] >= 10)
        
        criterios_cumplidos = sum(criterios)
        viable = criterios_cumplidos >= 3  # Al menos 3 de 4 criterios
        
        return viable, criterios_cumplidos


# ============================================
# FUNCIÓN PRINCIPAL SIMPLIFICADA
# ============================================

def main():
    print("🚀 SISTEMA DE TRADING OPTIMIZADO - VERSIÓN SIMPLIFICADA")
    print("=" * 80)
    print("NOTA: Esta versión tiene filtros relajados para testing inicial")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración temporal:")
    print(f"  Actual: {fechas['actual'].date()}")
    print(f"  Entrenamiento desde: {fechas['inicio_entrenamiento'].date()}")
    print(f"  Backtest desde: {fechas['inicio_backtest'].date()}")
    print(f"  Intervalo: {TradingConfig.INTERVALO}")
    print(f"  Horizontes: {TradingConfig.HORIZONTES} horas")
    
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
        print(f"  Criterios cumplidos: {criterios}/4")
        print(f"  Viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest
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
            print(f"    {ticker}: Operaciones={m['n_operaciones']}, "
                  f"Win rate={m['tasa_exito']:.2%}, "
                  f"Retorno={m['retorno_total']:.2%}, "
                  f"PF={m['profit_factor']:.2f}")
    
    return resultados_globales


if __name__ == "__main__":
    main()
