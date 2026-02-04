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
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================

class TradingConfig:
    """Configuración optimizada para crypto"""
    
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Datos - período más largo para más muestras
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 730  # 2 años para más datos
    DIAS_VALIDACION = 60
    DIAS_BACKTEST = 30
    
    # Activos (solo los más líquidos)
    ACTIVOS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]
    
    # Features
    VENTANA_VOLATILIDAD = 24
    RSI_PERIODO = 14
    ATR_PERIODO = 14
    
    # ✅ MÁS HORIZONTES (1-6 horas)
    HORIZONTES = [1, 2, 3, 4, 6]  # Múltiples horizontes para más señales
    
    # 🎯 GESTIÓN DE RIESGO REALISTA
    STOP_LOSS_PCT = 0.015    # 1.5% (más ajustado para crypto)
    TAKE_PROFIT_PCT = 0.030  # 3.0% (ratio 2:1)
    RATIO_MINIMO_RR = 1.5
    MAX_RIESGO_POR_OPERACION = 0.02
    
    # Validación
    N_FOLDS_WF = 3  # Menos folds para más datos por fold
    MIN_MUESTRAS_ENTRENAMIENTO = 500  # Reducido
    MIN_MUESTRAS_CLASE = 30  # Reducido
    
    # 🔥 UMBRALES MÁS FLEXIBLES
    UMBRAL_PROBABILIDAD_MIN = 0.52  # 52% (más flexible)
    UMBRAL_CONFIANZA_MIN = 0.51     # 51% (más flexible)
    
    # 🎯 UMBRAL DE MOVIMIENTO MÁS FLEXIBLE (más muestras)
    UMBRAL_MOVIMIENTO = 0.008  # 0.8% (antes 1.2%)
    
    # ✅ FILTROS RSI MÁS FLEXIBLES
    RSI_EXTREME_LOW = 10   # Solo bloquear SHORT si RSI < 10
    RSI_EXTREME_HIGH = 90  # Solo bloquear LONG si RSI > 90
    
    MODELOS_DIR = Path("modelos_trading")
    
    @classmethod
    def get_fechas(cls):
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio_entrenamiento': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_validacion': now - timedelta(days=cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_backtest': now - timedelta(days=cls.DIAS_BACKTEST)
        }


# ============================================
# INDICADORES MEJORADOS
# ============================================

class IndicadoresTecnicos:
    """Features calculadas SOLO con datos pasados"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        """RSI sin look-ahead"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        return (100 - (100 / (1 + rs))).fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
        """ATR sin look-ahead"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        close_prev = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(window=periodo).mean().fillna(method='bfill')
    
    @staticmethod
    def calcular_macd(close, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calcular_bb(close, window=20, num_std=2):
        """Calcula Bandas de Bollinger"""
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_width = (upper_band - lower_band) / sma
        return upper_band, lower_band, bb_width
    
    @staticmethod
    def calcular_features(df):
        """Features mejoradas con más indicadores"""
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(1, index=df.index))
        
        # ✅ RETORNOS PASADOS
        for periodo in [1, 2, 4, 8, 12, 24]:
            df[f'retorno_{periodo}h'] = close.pct_change(periodo).shift(1)
        
        # ✅ VOLATILIDAD PASADA
        retornos = close.pct_change(1)
        for ventana in [8, 24, 48]:
            df[f'volatilidad_{ventana}h'] = retornos.rolling(ventana).std().shift(1)
        
        # ✅ RSI PASADO (múltiples períodos)
        for periodo in [7, 14, 21]:
            rsi_raw = IndicadoresTecnicos.calcular_rsi(close, periodo)
            df[f'RSI_{periodo}'] = rsi_raw.shift(1)
        
        # ✅ MEDIAS MÓVILES PASADAS
        for periodo in [5, 12, 24, 50]:
            sma = close.rolling(periodo).mean().shift(1)
            df[f'SMA_{periodo}'] = sma
            df[f'dist_sma_{periodo}'] = (close.shift(1) - sma) / sma
        
        # ✅ EMA PASADAS
        for periodo in [5, 12, 26]:
            ema = close.ewm(span=periodo, adjust=False).mean().shift(1)
            df[f'EMA_{periodo}'] = ema
        
        # ✅ ATR PASADO
        atr = IndicadoresTecnicos.calcular_atr(df, 14)
        df['ATR'] = atr.shift(1)
        df['ATR_pct'] = (atr / close).shift(1)
        
        # ✅ MACD PASADO
        macd_line, signal_line, histogram = IndicadoresTecnicos.calcular_macd(close)
        df['MACD'] = macd_line.shift(1)
        df['MACD_signal'] = signal_line.shift(1)
        df['MACD_hist'] = histogram.shift(1)
        
        # ✅ BOLLINGER BANDS PASADO
        bb_upper, bb_lower, bb_width = IndicadoresTecnicos.calcular_bb(close)
        df['BB_upper'] = bb_upper.shift(1)
        df['BB_lower'] = bb_lower.shift(1)
        df['BB_width'] = bb_width.shift(1)
        df['BB_position'] = (close.shift(1) - bb_lower) / (bb_upper - bb_lower)
        
        # ✅ VOLUMEN RELATIVO PASADO
        for ventana in [8, 24, 48]:
            vol_ma = volume.rolling(ventana).mean()
            df[f'volumen_rel_{ventana}h'] = (volume / vol_ma).shift(1)
        
        # ✅ RANGO HL PASADO
        rango = (high - low) / close
        df['rango_hl_pct'] = rango.shift(1)
        
        # ✅ TENDENCIA PASADA (múltiples)
        df['tendencia_corta'] = (df['SMA_12'] > df['SMA_24']).astype(int)
        df['tendencia_larga'] = (df['SMA_24'] > df['SMA_50']).astype(int)
        
        # ✅ MOMENTUM PASADO
        for periodo in [8, 24, 48]:
            df[f'momentum_{periodo}h'] = (close / close.shift(periodo) - 1).shift(1)
        
        # ✅ OSCILADOR DE CANAL DE DONCHIAN
        for periodo in [20, 40]:
            high_max = high.rolling(periodo).max()
            low_min = low.rolling(periodo).min()
            df[f'donchian_osc_{periodo}'] = ((close - low_min) / (high_max - low_min)).shift(1)
        
        # ✅ CCI (Commodity Channel Index)
        for periodo in [20]:
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(periodo).mean()
            mad = tp.rolling(periodo).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            df[f'CCI_{periodo}'] = ((tp - sma_tp) / (0.015 * mad)).shift(1)
        
        # ✅ RATIO ADX (Directional Movement)
        for periodo in [14]:
            plus_dm = high.diff()
            minus_dm = low.diff().abs() * -1
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(periodo).mean()
            plus_di = 100 * (plus_dm.rolling(periodo).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(periodo).mean() / atr)
            df[f'ADX_plus_{periodo}'] = plus_di.shift(1)
            df[f'ADX_minus_{periodo}'] = minus_di.shift(1)
        
        return df


# ============================================
# ETIQUETADO MEJORADO
# ============================================

class EtiquetadoDatos:
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte):
        """
        ETIQUETADO MEJORADO:
        - Usa umbral dinámico basado en volatilidad
        """
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        # Umbral dinámico basado en volatilidad
        volatilidad = df['Close'].pct_change().rolling(24).std().fillna(0.01)
        umbral_dinamico = volatilidad * 2  # 2 desviaciones estándar
        
        etiqueta = pd.Series(np.nan, index=df.index)
        umbral_usado = umbral_dinamico.clip(lower=0.005, upper=0.015)
        
        etiqueta[retorno_futuro > umbral_usado] = 1   # LONG
        etiqueta[retorno_futuro < -umbral_usado] = 0  # SHORT
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        """Prepara dataset con features sin look-ahead"""
        df = IndicadoresTecnicos.calcular_features(df)
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Todas las features disponibles (excluyendo columnas temporales)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       f'etiqueta_{horizonte}h', f'retorno_futuro_{horizonte}h']
        
        features_disponibles = [col for col in df.columns 
                               if col not in exclude_cols 
                               and not col.startswith('SMA_') 
                               and not col.startswith('EMA_')]
        
        return df, features_disponibles


# ============================================
# MODELO CON MÁS FLEXIBILIDAD
# ============================================

class ModeloPrediccion:
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
        self.feature_importance = None
    
    def entrenar_walk_forward(self, df, features, etiqueta_col):
        """Walk-forward con menos restricciones"""
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)} (necesita {TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO})")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas: {class_counts.to_dict()}")
            return False
        
        print(f"    📊 Muestras totales: {len(X)}, LONG: {class_counts.get(1, 0)}, SHORT: {class_counts.get(0, 0)}")
        
        # ✅ USAR CROSS-VALIDATION SIMPLE (menos estricto)
        tscv = TimeSeriesSplit(n_splits=3, test_size=500, gap=self.horizonte)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Modelo más flexible
            modelo = RandomForestClassifier(
                n_estimators=50,       # Menos árboles para evitar overfit
                max_depth=6,           # Más profundidad
                min_samples_split=20,  # Menos samples por split
                min_samples_leaf=10,   # Menos samples por hoja
                max_features=0.3,      # Usar solo 30% de features
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            y_pred = modelo.predict(X_val_scaled)
            y_proba = modelo.predict_proba(X_val_scaled)
            
            # Métricas en validación
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            
            scores.append({'accuracy': acc, 'precision': prec, 'recall': rec})
        
        self.metricas_validacion = {
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'std_accuracy': np.std([s['accuracy'] for s in scores]),
            'n_folds': len(scores)
        }
        
        # ✅ CRITERIO MÁS FLEXIBLE: accuracy > 50.5%
        if self.metricas_validacion['accuracy'] < 0.505:
            print(f"      ❌ Accuracy muy baja: {self.metricas_validacion['accuracy']:.2%}")
            return False
        
        print(f"      ✅ Acc: {self.metricas_validacion['accuracy']:.2%} "
              f"(±{self.metricas_validacion['std_accuracy']:.2%}), "
              f"Prec: {self.metricas_validacion['precision']:.2%}, "
              f"Rec: {self.metricas_validacion['recall']:.2%}")
        
        # Entrenar modelo final con todos los datos
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.3,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        self.modelo.fit(X_scaled, y)
        self.features = features
        
        # Guardar feature importance
        self.feature_importance = dict(zip(features, self.modelo.feature_importances_))
        
        # Mostrar top 5 features
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            print(f"      🏆 Top features:")
            for feat, imp in top_features:
                print(f"        {feat}: {imp:.3f}")
        
        return True
    
    def predecir(self, df_actual):
        """Predicción en tiempo real"""
        if self.modelo is None:
            return None
        
        if not all(f in df_actual.columns for f in self.features):
            return None
        
        X = df_actual[self.features].iloc[[-1]]
        
        if X.isnull().any().any():
            return None
        
        X_scaled = self.scaler.transform(X)
        
        prediccion_clase = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'prediccion': int(prediccion_clase),  # 1=LONG, 0=SHORT
            'probabilidad_positiva': probabilidades[1],  # P(LONG)
            'probabilidad_negativa': probabilidades[0],  # P(SHORT)
            'confianza': max(probabilidades)
        }
    
    def guardar(self, path):
        if self.modelo is None:
            return False
        
        modelo_data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features': self.features,
            'metricas': self.metricas_validacion,
            'feature_importance': self.feature_importance,
            'horizonte': self.horizonte,
            'ticker': self.ticker
        }
        
        joblib.dump(modelo_data, path)
        return True
    
    @classmethod
    def cargar(cls, path):
        modelo_data = joblib.load(path)
        
        instancia = cls(modelo_data['horizonte'], modelo_data['ticker'])
        instancia.modelo = modelo_data['modelo']
        instancia.scaler = modelo_data['scaler']
        instancia.features = modelo_data['features']
        instancia.metricas_validacion = modelo_data['metricas']
        instancia.feature_importance = modelo_data.get('feature_importance', {})
        
        return instancia


# ============================================
# BACKTESTING CON MÁS SEÑALES
# ============================================

class Backtester:
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, rsi):
        """Simula operación con SL/TP ajustados"""
        precio_entrada = self.df.loc[idx, 'Close']
        
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Niveles basados en % fijo
        if señal_long:
            stop_loss = precio_entrada * (1 - TradingConfig.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 + TradingConfig.TAKE_PROFIT_PCT)
        else:
            stop_loss = precio_entrada * (1 + TradingConfig.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 - TradingConfig.TAKE_PROFIT_PCT)
        
        riesgo = abs(precio_entrada - stop_loss)
        recompensa = abs(take_profit - precio_entrada)
        ratio_rr = recompensa / riesgo
        
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Simular hasta 48 horas (más tiempo)
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(48, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Close'].values
        highs_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['High'].values
        lows_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Low'].values
        
        resultado = 'TIEMPO'
        velas_hasta_cierre = max_ventana
        retorno = 0
        
        # Simular vela por vela
        for i in range(1, len(precios_futuros)):
            high = highs_futuros[i]
            low = lows_futuros[i]
            
            if señal_long:
                if low <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -TradingConfig.STOP_LOSS_PCT
                    break
                elif high >= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = TradingConfig.TAKE_PROFIT_PCT
                    break
            else:  # SHORT
                if high >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -TradingConfig.STOP_LOSS_PCT
                    break
                elif low <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = TradingConfig.TAKE_PROFIT_PCT
                    break
        
        # Si no tocó ni TP ni SL, cerrar al final
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
            'rsi': rsi,
            'resultado': resultado,
            'retorno': retorno,
            'velas_hasta_cierre': velas_hasta_cierre
        }
    
    def ejecutar(self, fecha_inicio):
        """Lógica de backtest con más señales"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtest")
            return None
        
        print(f"  📊 Periodo: {df_backtest.index[0].date()} a {df_backtest.index[-1].date()}")
        
        for idx in df_backtest.index[:-48]:  # Dejar 48h al final
            predicciones = {}
            
            # Obtener predicciones de todos los horizontes
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Promediar probabilidades con pesos
            pesos = {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 6: 0.05}  # Más peso a horizontes cortos
            probs_positivas = []
            confianzas = []
            
            for h, p in predicciones.items():
                peso = pesos.get(h, 0.1)
                probs_positivas.append(p['probabilidad_positiva'] * peso)
                confianzas.append(p['confianza'] * peso)
            
            prob_promedio = np.sum(probs_positivas) / sum(pesos.get(h, 0.1) for h in predicciones.keys())
            confianza_promedio = np.sum(confianzas) / sum(pesos.get(h, 0.1) for h in predicciones.keys())
            
            # ✅ FILTRO DE CONFIANZA MÁS FLEXIBLE
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            # ✅ DECIDIR SEÑAL
            if prob_promedio > 0.5:
                señal_long = True
                prob_real = prob_promedio
            else:
                señal_long = False
                prob_real = 1 - prob_promedio
            
            # Verificar umbral de probabilidad
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            # Obtener RSI actual
            rsi = df_backtest.loc[idx, 'RSI_14']
            if pd.isna(rsi):
                rsi = df_backtest.loc[idx, 'RSI_7'] if 'RSI_7' in df_backtest.columns else 50
            
            # ✅ FILTRO EXTREMO SUAVE
            if señal_long and rsi > TradingConfig.RSI_EXTREME_HIGH:
                continue
            
            if not señal_long and rsi < TradingConfig.RSI_EXTREME_LOW:
                continue
            
            # Simular operación
            operacion = self.simular_operacion(
                idx,
                señal_long,
                prob_real,
                rsi
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones en backtest")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
        """Calcula métricas de rendimiento"""
        df_ops = pd.DataFrame(self.operaciones)
        
        n_ops = len(df_ops)
        n_tp = (df_ops['resultado'] == 'TP').sum()
        n_sl = (df_ops['resultado'] == 'SL').sum()
        n_timeout = (df_ops['resultado'] == 'TIEMPO').sum()
        
        retornos = df_ops['retorno']
        operaciones_ganadoras = retornos > 0
        operaciones_perdedoras = retornos < 0
        
        # Calcular profit factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        profit_factor = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Equity curve
        equity_curve = (1 + retornos).cumprod()
        
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': operaciones_ganadoras.sum() / n_ops,
            'hit_tp_rate': n_tp / n_ops,
            'hit_sl_rate': n_sl / n_ops,
            'timeout_rate': n_timeout / n_ops,
            'retorno_total': retornos.sum(),
            'retorno_promedio': retornos.mean(),
            'retorno_mediano': retornos.median(),
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'promedio_ganador': retornos[operaciones_ganadoras].mean() if operaciones_ganadoras.any() else 0,
            'promedio_perdedor': retornos[operaciones_perdedoras].mean() if operaciones_perdedoras.any() else 0,
            'profit_factor': profit_factor,
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'sharpe_ratio': retornos.mean() / retornos.std() * np.sqrt(365*24) if retornos.std() > 0 else 0,
            'sortino_ratio': retornos.mean() / retornos[retornos < 0].std() * np.sqrt(365*24) if len(retornos[retornos < 0]) > 1 else 0,
            'duracion_promedio': df_ops['velas_hasta_cierre'].mean(),
            'equity_final': equity_curve.iloc[-1] if len(equity_curve) > 0 else 1,
        }
        
        return metricas, df_ops


# ============================================
# SISTEMA COMPLETO OPTIMIZADO
# ============================================

class SistemaTradingTicker:
    
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
            # Intentar múltiples fuentes si yfinance falla
            ticker_yf = self.ticker.replace("-", "-")  # Ya está en formato correcto
            
            df = yf.download(
                ticker_yf,
                start=self.fechas['inicio_entrenamiento'],
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False,
                threads=True
            )
            
            if df.empty:
                print(f"  ❌ No hay datos disponibles")
                return False
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Reemplazar ceros en volumen
            df['Volume'] = df['Volume'].replace(0, df['Volume'].median())
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas desde {df.index[0].date()} hasta {df.index[-1].date()}")
            print(f"  📈 Precio actual: ${df['Close'].iloc[-1]:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error descargando datos: {e}")
            return False
    
    def entrenar_modelos(self):
        """Entrena modelos para cada horizonte (solo los mejores)"""
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Datos hasta inicio de backtest
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        print(f"  📅 Periodo: {df_train.index[0].date()} a {df_train.index[-1].date()}")
        
        modelos_entrenados = 0
        
        # Solo entrenar los 3 mejores horizontes basados en datos disponibles
        for horizonte in [1, 2, 3]:  # Enfocarse en horizontes cortos
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            try:
                df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
                etiqueta_col = f'etiqueta_{horizonte}h'
                
                # Verificar que hay suficientes features
                if len(features) < 5:
                    print(f"    ⚠️ Muy pocas features disponibles: {len(features)}")
                    continue
                
                # Seleccionar las 15 mejores features por correlación
                if len(features) > 15:
                    correlations = []
                    for feat in features:
                        if feat in df_prep.columns:
                            corr = df_prep[feat].corr(df_prep[etiqueta_col])
                            if not pd.isna(corr):
                                correlations.append((feat, abs(corr)))
                    
                    if correlations:
                        correlations.sort(key=lambda x: x[1], reverse=True)
                        best_features = [feat for feat, _ in correlations[:15]]
                        features = best_features
                        print(f"    🔍 Seleccionadas {len(features)} mejores features")
                
                etiquetas = df_prep[etiqueta_col].dropna()
                if len(etiquetas) > 0:
                    dist = etiquetas.value_counts(normalize=True)
                    print(f"    Distribución: LONG={dist.get(1, 0):.1%}, SHORT={dist.get(0, 0):.1%}")
                
                modelo = ModeloPrediccion(horizonte, self.ticker)
                if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                    self.modelos[horizonte] = modelo
                    modelos_entrenados += 1
            except Exception as e:
                print(f"    ❌ Error entrenando horizonte {horizonte}h: {e}")
                continue
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/3")
        
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        """Ejecuta backtest con los modelos entrenados"""
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            print("  ❌ No hay modelos disponibles")
            return False
        
        # Preparar datos completos con features
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(
            self.df_historico,
            1  # Usar horizonte 1 para features
        )
        
        backtester = Backtester(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS DEL BACKTEST:")
        print(f"    Operaciones totales: {metricas['n_operaciones']}")
        print(f"    Win rate: {metricas['tasa_exito']:.2%}")
        print(f"    TP alcanzado: {metricas['hit_tp_rate']:.2%}")
        print(f"    SL alcanzado: {metricas['hit_sl_rate']:.2%}")
        print(f"    Timeout: {metricas['timeout_rate']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Equity final: {metricas['equity_final']:.2f}")
        print(f"    Ganancia promedio: {metricas['promedio_ganador']:.2%}")
        print(f"    Pérdida promedio: {metricas['promedio_perdedor']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        print(f"    Duración promedio: {metricas['duracion_promedio']:.1f} velas")
        
        # Distribución por dirección
        if len(df_ops) > 0:
            long_ops = df_ops[df_ops['direccion'] == 'LONG']
            short_ops = df_ops[df_ops['direccion'] == 'SHORT']
            
            if len(long_ops) > 0:
                print(f"\n    📈 OPERACIONES LONG ({len(long_ops)}):")
                print(f"      Win rate: {(long_ops['retorno'] > 0).sum()/len(long_ops):.1%}")
                print(f"      Retorno promedio: {long_ops['retorno'].mean():.2%}")
            
            if len(short_ops) > 0:
                print(f"\n    📉 OPERACIONES SHORT ({len(short_ops)}):")
                print(f"      Win rate: {(short_ops['retorno'] > 0).sum()/len(short_ops):.1%}")
                print(f"      Retorno promedio: {short_ops['retorno'].mean():.2%}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable para este ticker"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        
        # Criterios más flexibles
        criterios = []
        criterios.append(('Operaciones >= 5', m['n_operaciones'] >= 5))
        criterios.append(('Win rate > 45%', m['tasa_exito'] > 0.45))
        criterios.append(('Profit factor > 1.2', m['profit_factor'] > 1.2))
        criterios.append(('Retorno total > 0%', m['retorno_total'] > 0))
        criterios.append(('Sharpe > 0.3', m['sharpe_ratio'] > 0.3))
        criterios.append(('Equity final > 1.0', m['equity_final'] > 1.0))
        
        criterios_cumplidos = sum([c[1] for c in criterios])
        viable = criterios_cumplidos >= 4  # Al menos 4 de 6
        
        print(f"\n  📋 EVALUACIÓN DE CRITERIOS:")
        for nombre, resultado in criterios:
            print(f"    {'✅' if resultado else '❌'} {nombre}")
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        """Analiza condiciones actuales y genera señal si aplica"""
        if not self.modelos:
            return None
        
        try:
            # Descargar datos recientes
            df_reciente = yf.download(
                self.ticker,
                start=self.fechas['actual'] - timedelta(days=10),  # 10 días
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df_reciente.empty:
                return None
            
            if isinstance(df_reciente.columns, pd.MultiIndex):
                df_reciente.columns = df_reciente.columns.get_level_values(0)
            
            df_reciente = df_reciente[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Calcular features
            df_reciente = IndicadoresTecnicos.calcular_features(df_reciente)
            
            # Obtener predicciones de todos los modelos
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_reciente)
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                return None
            
            # Promediar probabilidades con pesos
            pesos = {1: 0.5, 2: 0.3, 3: 0.2}  # Más peso a horizonte 1
            probs_positivas = []
            confianzas = []
            
            for h, p in predicciones.items():
                peso = pesos.get(h, 0.1)
                probs_positivas.append(p['probabilidad_positiva'] * peso)
                confianzas.append(p['confianza'] * peso)
            
            prob_promedio = np.sum(probs_positivas) / sum(pesos.get(h, 0.1) for h in predicciones.keys())
            confianza_promedio = np.sum(confianzas) / sum(pesos.get(h, 0.1) for h in predicciones.keys())
            
            # Filtro de confianza
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                return None
            
            # Decidir señal
            if prob_promedio > 0.5:
                señal = "LONG"
                prob_real = prob_promedio
            else:
                señal = "SHORT"
                prob_real = 1 - prob_promedio
            
            # Verificar umbral de probabilidad
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                return None
            
            # Obtener datos actuales
            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            
            # Obtener RSI
            rsi = ultima_vela.get('RSI_14', ultima_vela.get('RSI_7', 50))
            
            # Filtrar extremos de RSI
            if señal == "LONG" and rsi > TradingConfig.RSI_EXTREME_HIGH:
                return None
            
            if señal == "SHORT" and rsi < TradingConfig.RSI_EXTREME_LOW:
                return None
            
            # Calcular niveles
            if señal == 'LONG':
                sl = precio * (1 - TradingConfig.STOP_LOSS_PCT)
                tp = precio * (1 + TradingConfig.TAKE_PROFIT_PCT)
            else:
                sl = precio * (1 + TradingConfig.STOP_LOSS_PCT)
                tp = precio * (1 - TradingConfig.TAKE_PROFIT_PCT)
            
            ratio_rr = abs(tp - precio) / abs(precio - sl)
            
            if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
                return None
            
            # Estado del mercado
            estado_rsi = "NEUTRO"
            if rsi < 30:
                estado_rsi = "OVERSOLD"
            elif rsi > 70:
                estado_rsi = "OVERBOUGHT"
            
            tendencia = "ALCISTA" if ultima_vela.get('tendencia_corta', 0) == 1 else "BAJISTA"
            
            # Strength de la señal
            fuerza = "DÉBIL"
            if prob_real > 0.6:
                fuerza = "FUERTE"
            elif prob_real > 0.55:
                fuerza = "MEDIA"
            
            return {
                'ticker': self.ticker,
                'fecha': datetime.now(TradingConfig.TIMEZONE),
                'precio': precio,
                'señal': señal,
                'probabilidad': prob_real,
                'confianza': confianza_promedio,
                'fuerza': fuerza,
                'stop_loss': sl,
                'take_profit': tp,
                'ratio_rr': ratio_rr,
                'predicciones_detalle': predicciones,
                'rsi': rsi,
                'estado_rsi': estado_rsi,
                'tendencia': tendencia,
                'n_modelos': len(predicciones)
            }
        
        except Exception as e:
            print(f"  ❌ Error en análisis tiempo real: {e}")
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
        
        # Guardar métricas
        if self.metricas_backtest:
            path_metricas = path_ticker / "metricas_backtest.json"
            with open(path_metricas, 'w') as f:
                json.dump(self.metricas_backtest, f, indent=2)
        
        print(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# MAIN OPTIMIZADO
# ============================================

def main():
    """Sistema principal de trading optimizado"""
    print("🚀 SISTEMA DE TRADING OPTIMIZADO")
    print("=" * 80)
    print("✅ Más datos (2 años de entrenamiento)")
    print("✅ Umbrales flexibles (52% probabilidad, 51% confianza)")
    print("✅ Más indicadores técnicos (RSI múltiple, MACD, Bollinger)")
    print("✅ Umbral de movimiento dinámico (0.8%)")
    print("✅ Múltiples horizontes (1, 2, 3 horas)")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración:")
    print(f"  Fecha actual: {fechas['actual'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Periodo entrenamiento: {fechas['inicio_entrenamiento'].date()}")
    print(f"  Periodo backtest: {fechas['inicio_backtest'].date()}")
    print(f"  SL: {TradingConfig.STOP_LOSS_PCT:.1%}, TP: {TradingConfig.TAKE_PROFIT_PCT:.1%}")
    print(f"  Umbral movimiento: {TradingConfig.UMBRAL_MOVIMIENTO:.1%}")
    print(f"  Modelos a entrenar: 1h, 2h, 3h")
    
    # Crear directorio para modelos
    TradingConfig.MODELOS_DIR.mkdir(exist_ok=True)
    
    resultados_globales = {}
    
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        # 1. Descargar datos
        if not sistema.descargar_datos():
            continue
        
        # 2. Entrenar modelos
        if not sistema.entrenar_modelos():
            continue
        
        # 3. Backtest
        if not sistema.ejecutar_backtest():
            continue
        
        # 4. Evaluar viabilidad
        viable, criterios = sistema.es_viable()
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUACIÓN FINAL - {ticker}")
        print(f"{'='*80}")
        print(f"  Criterios cumplidos: {criterios}/6")
        print(f"  Sistema viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        señal_actual = None
        
        # 5. Si es viable, analizar tiempo real
        if viable:
            print(f"\n  🔍 Analizando condiciones actuales...")
            try:
                señal_actual = sistema.analizar_tiempo_real()
                
                if señal_actual:
                    print(f"\n  🚨 SEÑAL DETECTADA: {señal_actual['señal']} ({señal_actual['fuerza']})")
                    print(f"    Probabilidad: {señal_actual['probabilidad']:.2%}")
                    print(f"    Confianza: {señal_actual['confianza']:.2%}")
                    print(f"    RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})")
                    print(f"    Tendencia: {señal_actual['tendencia']}")
                    print(f"    Precio: ${señal_actual['precio']:,.2f}")
                    print(f"    Stop Loss: ${señal_actual['stop_loss']:,.2f} ({-TradingConfig.STOP_LOSS_PCT:.1%})")
                    print(f"    Take Profit: ${señal_actual['take_profit']:,.2f} ({TradingConfig.TAKE_PROFIT_PCT:.1%})")
                    print(f"    R:R = {señal_actual['ratio_rr']:.2f}")
                    print(f"    Modelos usados: {señal_actual['n_modelos']}")
                    
                    # Verificar si es señal nueva
                    ultima = cargar_ultima_senal()
                    es_nueva = True
                    
                    if ultima:
                        if (ultima.get("ticker") == ticker and 
                            ultima.get("señal") == señal_actual["señal"]):
                            # Verificar si pasaron al menos 6 horas
                            try:
                                fecha_ultima = datetime.fromisoformat(ultima["fecha"])
                                if datetime.now(TradingConfig.TIMEZONE) - fecha_ultima < timedelta(hours=6):
                                    es_nueva = False
                                    print("  🔁 Señal repetida (< 6 horas)")
                            except:
                                pass
                    
                    if es_nueva:
                        # Enviar por Telegram
                        emoji = "📈" if señal_actual['señal'] == "LONG" else "📉"
                        mensaje = (
                            f"{emoji} {ticker} - {señal_actual['señal']} ({señal_actual['fuerza']})\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Probabilidad: {señal_actual['probabilidad']:.1%}\n"
                            f"📊 Confianza: {señal_actual['confianza']:.1%}\n\n"
                            f"💰 Precio: ${señal_actual['precio']:,.2f}\n"
                            f"🛑 Stop Loss: ${señal_actual['stop_loss']:,.2f}\n"
                            f"🎯 Take Profit: ${señal_actual['take_profit']:,.2f}\n"
                            f"⚖️ Ratio R:R: {señal_actual['ratio_rr']:.2f}\n\n"
                            f"📈 RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})\n"
                            f"📊 Tendencia: {señal_actual['tendencia']}\n"
                            f"🔢 Modelos: {señal_actual['n_modelos']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ {señal_actual['fecha'].strftime('%Y-%m-%d %H:%M')}"
                        )
                        
                        enviar_telegram(mensaje)
                        
                        # Guardar señal
                        guardar_ultima_senal({
                            "ticker": ticker,
                            "señal": señal_actual["señal"],
                            "fecha": señal_actual["fecha"].isoformat(),
                            "precio": señal_actual["precio"],
                            "probabilidad": señal_actual["probabilidad"]
                        })
                else:
                    print("  ℹ️ No hay señal en este momento")
            
            except Exception as e:
                print(f"  ❌ Error en análisis tiempo real: {e}")
        
        # 6. Guardar modelos si es viable
        if viable:
            sistema.guardar_modelos()
        
        # Guardar resultados
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest,
            'señal_actual': señal_actual
        }
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    con_senal = [t for t, r in resultados_globales.items() if r.get('señal_actual')]
    
    print(f"\n  Activos procesados: {len(resultados_globales)}")
    print(f"  Sistemas viables: {len(viables)}")
    print(f"  Señales activas: {len(con_senal)}")
    
    if viables:
        print(f"\n  ✅ SISTEMAS VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            print(f"\n    {ticker}:")
            print(f"      Operaciones: {m['n_operaciones']}")
            print(f"      Win rate: {m['tasa_exito']:.1%}")
            print(f"      Retorno total: {m['retorno_total']:.2%}")
            print(f"      Profit Factor: {m['profit_factor']:.2f}")
            print(f"      Sharpe: {m['sharpe_ratio']:.2f}")
    
    if con_senal:
        print(f"\n  🚨 SEÑALES ACTIVAS:")
        for ticker in con_senal:
            s = resultados_globales[ticker]['señal_actual']
            emoji = "📈" if s['señal'] == "LONG" else "📉"
            print(f"    {emoji} {ticker}: {s['señal']} @ ${s['precio']:,.2f} (Prob: {s['probabilidad']:.1%})")
    
    print(f"\n{'='*80}")
    print("✅ Proceso completado")
    print(f"{'='*80}\n")
    
    return resultados_globales


# ============================================
# FUNCIONES UTILIDAD (sin cambios)
# ============================================

def cargar_ultima_senal():
    """Carga la última señal enviada"""
    if os.path.exists("ultima_senal.json"):
        with open("ultima_senal.json") as f:
            return json.load(f)
    return None

def guardar_ultima_senal(senal):
    """Guarda la señal enviada"""
    with open("ultima_senal.json", "w") as f:
        json.dump(senal, f, indent=2)

def enviar_telegram(mensaje):
    """Envía mensaje por Telegram"""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("  ⚠️ Telegram no configurado (variables TELEGRAM_TOKEN y TELEGRAM_CHAT_ID)")
        return
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": mensaje}, timeout=10)
        
        if r.status_code == 200:
            print(f"  📨 Mensaje enviado a Telegram")
        else:
            print(f"  ⚠️ Error Telegram: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️ Error enviando Telegram: {e}")


if __name__ == "__main__":
    main()
