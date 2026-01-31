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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import joblib
from pathlib import Path
import json
import optuna
import talib as ta
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": mensaje})
        print(f"📨 Telegram: {r.status_code}")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")


# ============================================
# CONFIGURACIÓN AVANZADA
# ============================================

class TradingConfig:
    """Configuración avanzada del sistema"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos de tiempo
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 365  # 1 año
    DIAS_VALIDACION = 90      # 3 meses
    DIAS_BACKTEST = 90        # 3 meses
    
    # Activos
    ACTIVOS = ["BTC-USD"]
    
    # Parámetros técnicos
    HORIZONTES = [4, 8, 12, 24]  # Horas
    
    # Modelos a probar
    MODELOS = {
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42, max_iter=500)
    }
    
    # Espacio de búsqueda de hiperparámetros
    PARAM_GRIDS = {
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }
    
    # Validación
    N_FOLDS = 5
    TEST_SIZE = 0.2
    
    # Umbrales de trading
    UMBRAL_PROBABILIDAD = 0.55
    UMBRAL_CONFIANZA = 0.60
    RATIO_RR_MINIMO = 1.5
    
    # Gestión de riesgo
    SL_MULTIPLIER = 2.0
    TP_MULTIPLIER = 3.0
    MAX_RISK_PER_TRADE = 0.02
    
    @classmethod
    def get_fechas(cls):
        """Calcula fechas del sistema"""
        now = datetime.now(cls.TIMEZONE)
        inicio_backtest = now - timedelta(days=cls.DIAS_BACKTEST)
        inicio_validacion = inicio_backtest - timedelta(days=cls.DIAS_VALIDACION)
        inicio_entrenamiento = inicio_validacion - timedelta(days=cls.DIAS_ENTRENAMIENTO)
        
        return {
            'actual': now,
            'inicio_entrenamiento': inicio_entrenamiento,
            'inicio_validacion': inicio_validacion,
            'inicio_backtest': inicio_backtest
        }


# ============================================
# FEATURE ENGINEERING AVANZADO
# ============================================

class AdvancedFeatureEngineer:
    """Motor de features avanzadas con TA-Lib"""
    
    @staticmethod
    def calcular_features_avanzadas(df):
        """Calcula más de 50 features técnicas avanzadas"""
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # 1. Indicadores de momentum
        df['RSI'] = ta.RSI(close, timeperiod=14)
        df['MFI'] = ta.MFI(high, low, close, volume, timeperiod=14)
        df['STOCH_K'], df['STOCH_D'] = ta.STOCH(high, low, close)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(close)
        df['ADX'] = ta.ADX(high, low, close, timeperiod=14)
        df['CCI'] = ta.CCI(high, low, close, timeperiod=20)
        df['ROC'] = ta.ROC(close, timeperiod=10)
        df['WILLR'] = ta.WILLR(high, low, close, timeperiod=14)
        
        # 2. Medias móviles y tendencias
        df['SMA_10'] = ta.SMA(close, timeperiod=10)
        df['SMA_20'] = ta.SMA(close, timeperiod=20)
        df['SMA_50'] = ta.SMA(close, timeperiod=50)
        df['EMA_12'] = ta.EMA(close, timeperiod=12)
        df['EMA_26'] = ta.EMA(close, timeperiod=26)
        
        df['SMA_10_RATIO'] = close / df['SMA_10']
        df['SMA_20_RATIO'] = close / df['SMA_20']
        df['SMA_50_RATIO'] = close / df['SMA_50']
        
        # 3. Bandas de Bollinger
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = ta.BBANDS(close, timeperiod=20)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
        df['BB_PERCENT'] = (close - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        # 4. Volatilidad
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        df['NATR'] = ta.NATR(high, low, close, timeperiod=14)
        df['TRANGE'] = ta.TRANGE(high, low, close)
        
        # 5. Volume-based indicators
        df['OBV'] = ta.OBV(close, volume)
        df['AD'] = ta.AD(high, low, close, volume)
        df['ADOSC'] = ta.ADOSC(high, low, close, volume)
        df['VWAP'] = (df['Close'] * volume).cumsum() / volume.cumsum()
        
        # 6. Pattern recognition
        df['CDL2CROWS'] = ta.CDL2CROWS(open=df['Open'], high=high, low=low, close=close)
        df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(open=df['Open'], high=high, low=low, close=close)
        df['CDL3INSIDE'] = ta.CDL3INSIDE(open=df['Open'], high=high, low=low, close=close)
        df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(open=df['Open'], high=high, low=low, close=close)
        
        # 7. Statistical features
        df['RETURN_1H'] = df['Close'].pct_change(1)
        df['RETURN_4H'] = df['Close'].pct_change(4)
        df['RETURN_12H'] = df['Close'].pct_change(12)
        df['RETURN_24H'] = df['Close'].pct_change(24)
        
        df['VOLATILITY_24H'] = df['RETURN_1H'].rolling(24).std()
        df['VOLATILITY_72H'] = df['RETURN_1H'].rolling(72).std()
        
        # 8. Rolling statistics
        for window in [8, 24, 72]:
            df[f'ROLL_MEAN_{window}'] = df['Close'].rolling(window).mean()
            df[f'ROLL_STD_{window}'] = df['Close'].rolling(window).std()
            df[f'ROLL_SKEW_{window}'] = df['Close'].rolling(window).skew()
            df[f'ROLL_KURT_{window}'] = df['Close'].rolling(window).kurt()
        
        # 9. Price position features
        df['HIGH_LOW_RATIO'] = high / low
        df['CLOSE_OPEN_RATIO'] = close / df['Open']
        df['TRUE_RANGE'] = ta.TRANGE(high, low, close)
        
        # 10. Volume features
        df['VOLUME_MA_10'] = volume.rolling(10).mean()
        df['VOLUME_MA_50'] = volume.rolling(50).mean()
        df['VOLUME_RATIO'] = volume / df['VOLUME_MA_10']
        df['VOLUME_AD'] = df['VOLUME_RATIO'] * df['RETURN_1H'].abs()
        
        # 11. Time-based features
        df['HOUR'] = df.index.hour
        df['DAY_OF_WEEK'] = df.index.dayofweek
        df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
        
        # 12. Market regime features
        df['TREND_STRENGTH'] = abs(df['Close'].pct_change(24).rolling(24).mean()) / (df['VOLATILITY_24H'] + 1e-10)
        df['REGIME'] = pd.qcut(df['VOLATILITY_24H'], q=4, labels=[1, 2, 3, 4])
        
        # 13. Mean reversion
        df['Z_SCORE_24H'] = (df['Close'] - df['Close'].rolling(24).mean()) / df['Close'].rolling(24).std()
        df['Z_SCORE_72H'] = (df['Close'] - df['Close'].rolling(72).mean()) / df['Close'].rolling(72).std()
        
        # 14. Support and Resistance
        for window in [20, 50, 100]:
            df[f'RESISTANCE_{window}'] = df['High'].rolling(window).max()
            df[f'SUPPORT_{window}'] = df['Low'].rolling(window).min()
            df[f'DIST_TO_RES_{window}'] = (df[f'RESISTANCE_{window}'] - close) / close
            df[f'DIST_TO_SUP_{window}'] = (close - df[f'SUPPORT_{window}']) / close
        
        # 15. Advanced momentum
        df['MOMENTUM_4H'] = ta.MOM(close, timeperiod=4)
        df['MOMENTUM_12H'] = ta.MOM(close, timeperiod=12)
        df['MOMENTUM_24H'] = ta.MOM(close, timeperiod=24)
        df['MOMENTUM_CHANGE'] = df['MOMENTUM_4H'].diff()
        
        # 16. Rate of Change features
        for period in [1, 4, 12, 24]:
            df[f'ROC_{period}'] = ta.ROC(close, timeperiod=period)
        
        # 17. Price derivatives
        df['PRICE_ACCELERATION'] = df['RETURN_1H'].diff()
        df['PRICE_JERK'] = df['PRICE_ACCELERATION'].diff()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df
    
    @staticmethod
    def get_feature_groups():
        """Retorna grupos de features para análisis"""
        return {
            'momentum': ['RSI', 'MFI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACD_HIST', 
                        'CCI', 'ROC', 'WILLR', 'MOMENTUM_4H', 'MOMENTUM_12H', 'MOMENTUM_24H'],
            'trend': ['ADX', 'SMA_10_RATIO', 'SMA_20_RATIO', 'SMA_50_RATIO', 
                     'EMA_12', 'EMA_26', 'TREND_STRENGTH'],
            'volatility': ['ATR', 'NATR', 'VOLATILITY_24H', 'VOLATILITY_72H', 
                          'BB_WIDTH', 'TRUE_RANGE'],
            'volume': ['OBV', 'AD', 'ADOSC', 'VOLUME_RATIO', 'VOLUME_AD',
                      'MFI', 'VWAP'],
            'price_position': ['BB_PERCENT', 'HIGH_LOW_RATIO', 'CLOSE_OPEN_RATIO',
                              'Z_SCORE_24H', 'Z_SCORE_72H'],
            'pattern': ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE'],
            'returns': ['RETURN_1H', 'RETURN_4H', 'RETURN_12H', 'RETURN_24H',
                       'ROC_1', 'ROC_4', 'ROC_12', 'ROC_24'],
            'support_resistance': ['DIST_TO_RES_20', 'DIST_TO_RES_50', 'DIST_TO_RES_100',
                                  'DIST_TO_SUP_20', 'DIST_TO_SUP_50', 'DIST_TO_SUP_100'],
            'time': ['HOUR', 'DAY_OF_WEEK', 'IS_WEEKEND'],
            'statistical': ['ROLL_SKEW_8', 'ROLL_SKEW_24', 'ROLL_SKEW_72',
                           'ROLL_KURT_8', 'ROLL_KURT_24', 'ROLL_KURT_72',
                           'PRICE_ACCELERATION', 'PRICE_JERK']
        }


# ============================================
# ETIQUETADO INTELIGENTE
# ============================================

class SmartLabeler:
    """Sistema de etiquetado inteligente"""
    
    @staticmethod
    def crear_etiquetas_multicriterio(df, horizonte):
        """
        Crea etiquetas usando múltiples criterios:
        1. Retorno futuro
        2. Volatilidad ajustada
        3. Tendencias del mercado
        4. Posición relativa
        """
        # Precio futuro
        precio_futuro = df['Close'].shift(-horizonte)
        retorno_futuro = (precio_futuro / df['Close']) - 1
        
        # 1. Criterio de retorno simple
        umbral_dinamico = df['VOLATILITY_24H'] * 1.5
        etiqueta_retorno = pd.Series(np.nan, index=df.index)
        etiqueta_retorno[retorno_futuro > umbral_dinamico] = 1
        etiqueta_retorno[retorno_futuro < -umbral_dinamico] = 0
        
        # 2. Criterio de posición en BB
        etiqueta_bb = pd.Series(0.5, index=df.index)  # Neutral por defecto
        etiqueta_bb[df['BB_PERCENT'] > 0.8] = 1  # Sobrecomprado -> posible short
        etiqueta_bb[df['BB_PERCENT'] < 0.2] = 0  # Sobreventado -> posible long
        
        # 3. Criterio de RSI
        etiqueta_rsi = pd.Series(0.5, index=df.index)
        etiqueta_rsi[df['RSI'] > 70] = 1  # Sobrecomprado
        etiqueta_rsi[df['RSI'] < 30] = 0  # Sobreventado
        
        # 4. Criterio de momentum
        etiqueta_mom = pd.Series(0.5, index=df.index)
        etiqueta_mom[df['MOMENTUM_12H'] > 0] = 1
        etiqueta_mom[df['MOMENTUM_12H'] < 0] = 0
        
        # Combinar criterios con pesos
        peso_retorno = 0.4
        peso_bb = 0.2
        peso_rsi = 0.2
        peso_mom = 0.2
        
        etiqueta_combinada = (
            etiqueta_retorno * peso_retorno +
            etiqueta_bb * peso_bb +
            etiqueta_rsi * peso_rsi +
            etiqueta_mom * peso_mom
        )
        
        # Convertir a binario
        etiqueta_final = pd.Series(np.nan, index=df.index)
        etiqueta_final[etiqueta_combinada > 0.5] = 1
        etiqueta_final[etiqueta_combinada < 0.5] = 0
        
        return etiqueta_final, retorno_futuro
    
    @staticmethod
    def crear_etiquetas_triples(df, horizonte):
        """Etiquetas de 3 clases: LONG, SHORT, HOLD"""
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        # Umbrales dinámicos
        umbral_superior = df['VOLATILITY_24H'] * 2.0
        umbral_inferior = df['VOLATILITY_24H'] * 2.0
        
        etiqueta = pd.Series(1, index=df.index)  # HOLD por defecto
        etiqueta[retorno_futuro > umbral_superior] = 2  # STRONG LONG
        etiqueta[retorno_futuro < -umbral_inferior] = 0  # STRONG SHORT
        
        return etiqueta, retorno_futuro


# ============================================
# OPTIMIZADOR DE HIPERPARÁMETROS
# ============================================

class HyperparameterOptimizer:
    """Optimizador avanzado de hiperparámetros"""
    
    def __init__(self, cv_splits=5):
        self.cv_splits = cv_splits
        self.best_params = {}
        self.best_scores = {}
        
    def optimize_model(self, X, y, modelo_nombre, modelo, param_grid):
        """Optimiza un modelo usando RandomizedSearchCV"""
        
        # TimeSeriesSplit para validación temporal
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # Randomized Search con métricas múltiples
        search = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=param_grid,
            n_iter=20,  # Número de combinaciones a probar
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        # Escalar datos
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Buscar mejores parámetros
        print(f"    🔍 Optimizando {modelo_nombre}...")
        search.fit(X_scaled, y)
        
        self.best_params[modelo_nombre] = search.best_params_
        self.best_scores[modelo_nombre] = search.best_score_
        
        print(f"      ✅ Mejores parámetros: {search.best_params_}")
        print(f"      📈 Mejor score: {search.best_score_:.4f}")
        
        return search.best_estimator_, scaler
    
    def optimize_with_optuna(self, X, y, modelo_nombre, n_trials=50):
        """Optimización con Optuna (más avanzada)"""
        
        def objective(trial):
            if modelo_nombre == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
                }
                modelo = XGBClassifier(**params, random_state=42, n_jobs=-1)
            
            elif modelo_nombre == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
                }
                modelo = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
            
            # Validación cruzada
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_val_scaled)
                scores.append(accuracy_score(y_val, y_pred))
            
            return np.mean(scores)
        
        # Crear estudio Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Mejores parámetros
        self.best_params[modelo_nombre] = study.best_params
        self.best_scores[modelo_nombre] = study.best_value
        
        # Entrenar modelo final con mejores parámetros
        if modelo_nombre == 'XGBoost':
            best_model = XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
        elif modelo_nombre == 'LightGBM':
            best_model = LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        return best_model, scaler


# ============================================
# ENSEMBLE MODEL
# ============================================

class AdvancedEnsemble:
    """Ensemble avanzado de modelos"""
    
    def __init__(self, modelos, scalers):
        self.modelos = modelos
        self.scalers = scalers
        self.pesos = None
        
    def calibrar_pesos(self, X_val, y_val):
        """Calibra pesos basados en rendimiento en validación"""
        accuracies = []
        
        for nombre, (modelo, scaler) in self.modelos.items():
            X_val_scaled = scaler.transform(X_val)
            y_pred = modelo.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)
        
        # Pesos proporcionales a accuracy
        self.pesos = np.array(accuracies) / sum(accuracies)
        return self.pesos
    
    def predecir(self, X):
        """Predicción ponderada del ensemble"""
        predicciones = []
        probabilidades = []
        
        for nombre, (modelo, scaler) in self.modelos.items():
            X_scaled = scaler.transform(X)
            pred = modelo.predict(X_scaled)
            proba = modelo.predict_proba(X_scaled)[:, 1]
            predicciones.append(pred)
            probabilidades.append(proba)
        
        # Promedio ponderado
        if self.pesos is not None:
            pred_final = np.average(predicciones, axis=0, weights=self.pesos)
            prob_final = np.average(probabilidades, axis=0, weights=self.pesos)
        else:
            pred_final = np.mean(predicciones, axis=0)
            prob_final = np.mean(probabilidades, axis=0)
        
        return {
            'prediccion': (pred_final > 0.5).astype(int),
            'probabilidad': prob_final,
            'confianza': np.abs(prob_final - 0.5) * 2,
            'consenso': np.std(predicciones, axis=0)  # Baja = alto consenso
        }


# ============================================
# BACKTESTING AVANZADO
# ============================================

class AdvancedBacktester:
    """Backtesting con múltiples estrategias y métricas"""
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos  # Dict de ensembles por horizonte
        self.ticker = ticker
        self.resultados = {}
        
    def probar_estrategias(self, fecha_inicio):
        """Prueba múltiples estrategias de trading"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        estrategias = {
            'basica': self._estrategia_basica,
            'conservadora': self._estrategia_conservadora,
            'agresiva': self._estrategia_agresiva,
            'consenso': self._estrategia_consenso
        }
        
        for nombre, estrategia in estrategias.items():
            print(f"\n  🧪 Probando estrategia: {nombre}")
            resultado = estrategia(df_backtest)
            if resultado:
                self.resultados[nombre] = resultado
        
        return self.resultados
    
    def _estrategia_basica(self, df):
        """Estrategia básica: Señal de un solo horizonte"""
        operaciones = []
        
        for i in range(24, len(df) - 24):
            idx = df.index[i]
            
            # Usar horizonte de 12h
            if 12 in self.modelos:
                ensemble = self.modelos[12]
                pred = ensemble.predecir(df.iloc[:i+1].tail(100))  # Últimas 100 velas
                
                if pred['confianza'][0] > TradingConfig.UMBRAL_CONFIANZA:
                    operacion = self._simular_operacion(df, i, pred['prediccion'][0], pred['probabilidad'][0])
                    if operacion:
                        operaciones.append(operacion)
        
        return self._calcular_metricas(operaciones)
    
    def _estrategia_conservadora(self, df):
        """Estrategia conservadora: Requiere confirmación múltiple"""
        operaciones = []
        
        for i in range(50, len(df) - 24):
            idx = df.index[i]
            
            predicciones = []
            for horizonte, ensemble in self.modelos.items():
                pred = ensemble.predecir(df.iloc[:i+1].tail(100))
                if pred['confianza'][0] > 0.65:  # Umbral alto
                    predicciones.append(pred['prediccion'][0])
            
            if len(predicciones) >= 2 and len(set(predicciones)) == 1:  # Consenso unánime
                operacion = self._simular_operacion(df, i, predicciones[0], 0.7)  # Alta confianza
                if operacion:
                    operaciones.append(operacion)
        
        return self._calcular_metricas(operaciones)
    
    def _estrategia_agresiva(self, df):
        """Estrategia agresiva: Más operaciones con filtros menos estrictos"""
        operaciones = []
        
        for i in range(12, len(df) - 12):
            idx = df.index[i]
            
            # Combinar predicciones de todos los horizontes
            probs = []
            for horizonte, ensemble in self.modelos.items():
                pred = ensemble.predecir(df.iloc[:i+1].tail(50))
                probs.append(pred['probabilidad'][0])
            
            prob_promedio = np.mean(probs)
            
            if prob_promedio > 0.6 or prob_promedio < 0.4:  # Fuerte señal en cualquier dirección
                direccion = 1 if prob_promedio > 0.5 else 0
                operacion = self._simular_operacion(df, i, direccion, abs(prob_promedio - 0.5) * 2)
                if operacion:
                    operaciones.append(operacion)
        
        return self._calcular_metricas(operaciones)
    
    def _estrategia_consenso(self, df):
        """Estrategia de consenso: Votación ponderada"""
        operaciones = []
        
        for i in range(36, len(df) - 24):
            idx = df.index[i]
            
            votos = []
            pesos = []
            
            for horizonte, ensemble in self.modelos.items():
                pred = ensemble.predecir(df.iloc[:i+1].tail(100))
                votos.append(pred['prediccion'][0])
                pesos.append(pred['confianza'][0])
            
            if len(votos) > 0:
                # Votación ponderada por confianza
                voto_ponderado = np.average(votos, weights=pesos)
                confianza_promedio = np.mean(pesos)
                
                if confianza_promedio > 0.6:
                    operacion = self._simular_operacion(df, i, 1 if voto_ponderado > 0.5 else 0, confianza_promedio)
                    if operacion:
                        operaciones.append(operacion)
        
        return self._calcular_metricas(operaciones)
    
    def _simular_operacion(self, df, idx_pos, direccion, confianza):
        """Simula una operación con gestión de riesgo avanzada"""
        entrada = df.iloc[idx_pos]
        precio = entrada['Close']
        atr = entrada.get('ATR', precio * 0.02)
        volatilidad = entrada.get('VOLATILITY_24H', 0.02)
        
        # Ajustar SL/TP basado en volatilidad
        sl_mult = TradingConfig.SL_MULTIPLIER * (1 + volatilidad * 10)
        tp_mult = TradingConfig.TP_MULTIPLIER * (1 + volatilidad * 5)
        
        if direccion == 1:  # LONG
            sl = precio * (1 - sl_mult * atr / precio)
            tp = precio * (1 + tp_mult * atr / precio)
        else:  # SHORT
            sl = precio * (1 + sl_mult * atr / precio)
            tp = precio * (1 - tp_mult * atr / precio)
        
        # Calcular ratio R:R
        riesgo = abs(precio - sl)
        recompensa = abs(tp - precio)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        if ratio_rr < TradingConfig.RATIO_RR_MINIMO:
            return None
        
        # Simular
        resultado, retorno, velas = self._simular_resultado(df, idx_pos, direccion, sl, tp)
        
        return {
            'fecha': df.index[idx_pos],
            'direccion': 'LONG' if direccion == 1 else 'SHORT',
            'precio': precio,
            'sl': sl,
            'tp': tp,
            'ratio_rr': ratio_rr,
            'confianza': confianza,
            'resultado': resultado,
            'retorno': retorno,
            'velas': velas,
            'volatilidad': volatilidad
        }
    
    def _simular_resultado(self, df, idx_pos, direccion, sl, tp):
        """Simula el resultado de una operación"""
        max_velas = 48
        
        for j in range(1, min(max_velas, len(df) - idx_pos - 1)):
            precio = df.iloc[idx_pos + j]['Close']
            
            if direccion == 1:  # LONG
                if precio >= tp:
                    return 'TP', (tp - df.iloc[idx_pos]['Close']) / df.iloc[idx_pos]['Close'], j
                elif precio <= sl:
                    return 'SL', (sl - df.iloc[idx_pos]['Close']) / df.iloc[idx_pos]['Close'], j
            else:  # SHORT
                if precio <= tp:
                    return 'TP', (df.iloc[idx_pos]['Close'] - tp) / df.iloc[idx_pos]['Close'], j
                elif precio >= sl:
                    return 'SL', (df.iloc[idx_pos]['Close'] - sl) / df.iloc[idx_pos]['Close'], j
        
        # Si no se activa SL/TP
        precio_final = df.iloc[idx_pos + min(max_velas - 1, len(df) - idx_pos - 2)]['Close']
        if direccion == 1:
            retorno = (precio_final - df.iloc[idx_pos]['Close']) / df.iloc[idx_pos]['Close']
        else:
            retorno = (df.iloc[idx_pos]['Close'] - precio_final) / df.iloc[idx_pos]['Close']
        
        return 'TIMEOUT', retorno, max_velas
    
    def _calcular_metricas(self, operaciones):
        """Calcula métricas avanzadas de rendimiento"""
        if not operaciones:
            return None
        
        df_ops = pd.DataFrame(operaciones)
        retornos = df_ops['retorno']
        ganadoras = retornos > 0
        
        # Métricas básicas
        n_ops = len(df_ops)
        win_rate = ganadoras.mean()
        retorno_total = retornos.sum()
        retorno_promedio = retornos.mean()
        
        # Drawdown
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Profit Factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        pf = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Sharpe Ratio
        sharpe = retornos.mean() / retornos.std() if retornos.std() > 0 else 0
        
        # Calmar Ratio
        calmar = retorno_total / abs(max_dd) if max_dd != 0 else np.inf
        
        # Expectancy
        avg_win = retornos[retornos > 0].mean() if len(retornos[retornos > 0]) > 0 else 0
        avg_loss = abs(retornos[retornos < 0].mean()) if len(retornos[retornos < 0]) > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Recovery Factor
        recovery = ganancias / abs(perdidas) if perdidas > 0 else np.inf
        
        # Análisis por tipo
        long_ops = df_ops[df_ops['direccion'] == 'LONG']
        short_ops = df_ops[df_ops['direccion'] == 'SHORT']
        
        win_rate_long = long_ops['retorno'].gt(0).mean() if len(long_ops) > 0 else 0
        win_rate_short = short_ops['retorno'].gt(0).mean() if len(short_ops) > 0 else 0
        
        return {
            'n_operaciones': n_ops,
            'win_rate': win_rate,
            'win_rate_long': win_rate_long,
            'win_rate_short': win_rate_short,
            'retorno_total': retorno_total,
            'retorno_promedio': retorno_promedio,
            'max_drawdown': max_dd,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'expectancy': expectancy,
            'recovery_factor': recovery,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': retornos.max(),
            'worst_trade': retornos.min(),
            'operaciones': df_ops
        }


# ============================================
# SISTEMA PRINCIPAL
# ============================================

class AdvancedTradingSystem:
    """Sistema de trading completo con optimización avanzada"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.fechas = TradingConfig.get_fechas()
        self.datos = None
        self.modelos = {}
        self.resultados = {}
        
    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de trading"""
        print(f"\n{'='*80}")
        print(f"🚀 SISTEMA AVANZADO DE TRADING - {self.ticker}")
        print(f"{'='*80}")
        
        # 1. Cargar datos
        if not self._cargar_datos():
            return False
        
        # 2. Procesar datos
        df_procesado = self._procesar_datos()
        
        # 3. Dividir en conjuntos
        splits = self._dividir_datos(df_procesado)
        
        # 4. Entrenar modelos para cada horizonte
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n🎯 ENTRENANDO PARA HORIZONTE {horizonte}h")
            print("-" * 60)
            
            # Crear etiquetas
            df_labeled, features = self._crear_dataset_con_etiquetas(splits['train_val'], horizonte)
            
            # Entrenar ensemble para este horizonte
            ensemble = self._entrenar_ensemble_para_horizonte(df_labeled, features, horizonte)
            if ensemble:
                self.modelos[horizonte] = ensemble
        
        if not self.modelos:
            print("❌ No se pudieron entrenar modelos")
            return False
        
        # 5. Backtesting avanzado
        print(f"\n🔬 BACKTESTING AVANZADO")
        print("-" * 60)
        
        backtester = AdvancedBacktester(splits['test'], self.modelos, self.ticker)
        resultados = backtester.probar_estrategias(self.fechas['inicio_backtest'])
        
        # 6. Analizar resultados
        self._analizar_resultados(resultados)
        
        # 7. Evaluación final
        self._evaluar_sistema()
        
        return True
    
    def _cargar_datos(self):
        """Carga datos históricos"""
        print("📥 Cargando datos...")
        
        try:
            # Descargar más datos de los necesarios para tener buffer
            start_date = self.fechas['inicio_entrenamiento'] - timedelta(days=100)
            
            df = yf.download(
                self.ticker,
                start=start_date,
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df.empty:
                print("❌ No hay datos disponibles")
                return False
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            self.datos = df
            
            print(f"✅ {len(df)} velas cargadas")
            print(f"📅 Rango: {df.index[0]} a {df.index[-1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def _procesar_datos(self):
        """Procesa los datos con features avanzadas"""
        print("🔄 Procesando datos con features avanzadas...")
        
        engineer = AdvancedFeatureEngineer()
        df_procesado = engineer.calcular_features_avanzadas(self.datos)
        
        # Eliminar primeras filas con NaN
        df_procesado = df_procesado.dropna()
        
        print(f"✅ Datos procesados: {len(df_procesado)} filas, {len(df_procesado.columns)} features")
        
        return df_procesado
    
    def _dividir_datos(self, df):
        """Divide los datos en conjuntos de entrenamiento, validación y test"""
        print("📊 Dividiendo datos...")
        
        # Fechas de corte
        fecha_test = self.fechas['inicio_backtest']
        fecha_val = self.fechas['inicio_validacion']
        
        splits = {
            'train': df[df.index < fecha_val],
            'val': df[(df.index >= fecha_val) & (df.index < fecha_test)],
            'test': df[df.index >= fecha_test]
        }
        
        print(f"  Train: {len(splits['train'])} velas ({splits['train'].index[0]} a {splits['train'].index[-1]})")
        print(f"  Val:   {len(splits['val'])} velas ({splits['val'].index[0]} a {splits['val'].index[-1]})")
        print(f"  Test:  {len(splits['test'])} velas ({splits['test'].index[0]} a {splits['test'].index[-1]})")
        
        # Combinar train y val para entrenamiento final
        splits['train_val'] = pd.concat([splits['train'], splits['val']])
        
        return splits
    
    def _crear_dataset_con_etiquetas(self, df, horizonte):
        """Crea dataset con etiquetas para un horizonte específico"""
        # Calcular retorno futuro
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        # Crear etiquetas binarias basadas en percentiles
        threshold_up = retorno_futuro.quantile(0.6)
        threshold_down = retorno_futuro.quantile(0.4)
        
        etiquetas = pd.Series(0.5, index=df.index)  # Neutral por defecto
        etiquetas[retorno_futuro >= threshold_up] = 1
        etiquetas[retorno_futuro <= threshold_down] = 0
        
        # Seleccionar features (excluir las que causan leakage)
        exclude_features = ['Close', 'High', 'Low', 'Open', 'Volume'] + \
                          [col for col in df.columns if 'RETURN_' in col or 'FUTURE' in col]
        
        features = [col for col in df.columns if col not in exclude_features and not col.startswith('label_')]
        
        # Filtrar filas con etiquetas válidas
        df_valid = df[etiquetas.notna()].copy()
        etiquetas_valid = etiquetas[etiquetas.notna()]
        
        print(f"  Dataset: {len(df_valid)} muestras, {len(features)} features")
        print(f"  Balance de clases: {etiquetas_valid.value_counts().to_dict()}")
        
        return df_valid[features], etiquetas_valid, features
    
    def _entrenar_ensemble_para_horizonte(self, df, etiquetas, features, horizonte):
        """Entrena un ensemble de modelos para un horizonte específico"""
        print(f"\n  🏗️  Construyendo ensemble para horizonte {horizonte}h")
        
        # Dividir en train y validation
        split_idx = int(len(df) * 0.8)
        X_train = df.iloc[:split_idx]
        y_train = etiquetas.iloc[:split_idx]
        X_val = df.iloc[split_idx:]
        y_val = etiquetas.iloc[split_idx:]
        
        modelos_entrenados = {}
        scalers = {}
        
        # Probar varios modelos
        modelos_a_probar = ['XGBoost', 'LightGBM', 'RandomForest']
        
        for modelo_nombre in modelos_a_probar:
            print(f"\n    🤖 Entrenando {modelo_nombre}...")
            
            try:
                # Optimizar hiperparámetros
                optimizer = HyperparameterOptimizer(cv_splits=3)
                modelo, scaler = optimizer.optimize_model(
                    X_train, y_train,
                    modelo_nombre,
                    TradingConfig.MODELOS[modelo_nombre],
                    TradingConfig.PARAM_GRIDS[modelo_nombre]
                )
                
                # Evaluar en validation
                X_val_scaled = scaler.transform(X_val)
                y_pred = modelo.predict(X_val_scaled)
                y_proba = modelo.predict_proba(X_val_scaled)[:, 1]
                
                acc = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                print(f"      📊 Validation Metrics:")
                print(f"        Accuracy:  {acc:.4f}")
                print(f"        Precision: {precision:.4f}")
                print(f"        Recall:    {recall:.4f}")
                print(f"        F1-Score:  {f1:.4f}")
                
                if acc > 0.55:  # Umbral mínimo
                    modelos_entrenados[modelo_nombre] = (modelo, scaler)
                    scalers[modelo_nombre] = scaler
                    
            except Exception as e:
                print(f"      ❌ Error entrenando {modelo_nombre}: {e}")
                continue
        
        if not modelos_entrenados:
            print("    ⚠️ No se pudo entrenar ningún modelo para este horizonte")
            return None
        
        # Crear ensemble
        print(f"\n    🎭 Creando ensemble con {len(modelos_entrenados)} modelos...")
        ensemble = AdvancedEnsemble(modelos_entrenados, scalers)
        
        # Calibrar pesos del ensemble
        X_val_scaled = None
        for nombre, (modelo, scaler) in modelos_entrenados.items():
            if X_val_scaled is None:
                X_val_scaled = scaler.transform(X_val)
        
        if X_val_scaled is not None:
            pesos = ensemble.calibrar_pesos(X_val_scaled, y_val)
            print(f"      📐 Pesos del ensemble: {dict(zip(modelos_entrenados.keys(), pesos))}")
        
        return ensemble
    
    def _analizar_resultados(self, resultados):
        """Analiza los resultados de todas las estrategias"""
        print(f"\n{'='*80}")
        print("📊 ANÁLISIS DE RESULTADOS")
        print(f"{'='*80}")
        
        if not resultados:
            print("❌ No hay resultados para analizar")
            return
        
        # Encontrar la mejor estrategia
        mejor_estrategia = None
        mejor_score = -np.inf
        
        for estrategia, metricas in resultados.items():
            if metricas:
                # Score compuesto
                score = (
                    metricas['win_rate'] * 0.3 +
                    metricas['profit_factor'] * 0.2 +
                    (1 - abs(metricas['max_drawdown'])) * 0.2 +
                    metricas['sharpe_ratio'] * 0.15 +
                    metricas['calmar_ratio'] * 0.15
                )
                
                print(f"\n📈 Estrategia: {estrategia.upper()}")
                print(f"  Score: {score:.4f}")
                print(f"  Operaciones: {metricas['n_operaciones']}")
                print(f"  Win Rate: {metricas['win_rate']:.2%}")
                print(f"  Retorno Total: {metricas['retorno_total']:.2%}")
                print(f"  Profit Factor: {metricas['profit_factor']:.2f}")
                print(f"  Max Drawdown: {metricas['max_drawdown']:.2%}")
                print(f"  Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
                print(f"  Expectancy: {metricas['expectancy']:.4f}")
                
                if score > mejor_score:
                    mejor_score = score
                    mejor_estrategia = estrategia
        
        if mejor_estrategia:
            print(f"\n🏆 MEJOR ESTRATEGIA: {mejor_estrategia.upper()}")
            print(f"   Score: {mejor_score:.4f}")
            self.resultados = resultados[mejor_estrategia]
    
    def _evaluar_sistema(self):
        """Evalúa si el sistema es viable"""
        if not self.resultados:
            print("❌ No hay resultados para evaluar")
            return False
        
        m = self.resultados
        
        criterios = {
            'Win Rate > 50%': m['win_rate'] > 0.50,
            'Profit Factor > 1.3': m['profit_factor'] > 1.3,
            'Max DD < 20%': abs(m['max_drawdown']) < 0.20,
            'Sharpe > 0.5': m['sharpe_ratio'] > 0.5,
            'Operaciones >= 15': m['n_operaciones'] >= 15,
            'Expectancy > 0': m['expectancy'] > 0,
            'Calmar > 1': m['calmar_ratio'] > 1,
            'Recovery > 1': m['recovery_factor'] > 1
        }
        
        cumplidos = sum(criterios.values())
        total = len(criterios)
        
        print(f"\n{'='*80}")
        print("📋 EVALUACIÓN DE VIABILIDAD")
        print(f"{'='*80}")
        
        for criterio, cumple in criterios.items():
            print(f"  {'✅' if cumple else '❌'} {criterio}")
        
        print(f"\n  Criterios cumplidos: {cumplidos}/{total}")
        
        viable = cumplidos >= 6  # Al menos 6 de 8 criterios
        
        if viable:
            print(f"\n🎉 SISTEMA VIABLE DETECTADO!")
            print("   Considerar implementación en tiempo real con monitoreo continuo")
            
            # Generar señal actual
            self._generar_senal_actual()
        else:
            print(f"\n⚠️ Sistema no viable en condiciones actuales")
            print("   Requiere optimización adicional")
        
        return viable
    
    def _generar_senal_actual(self):
        """Genera señal de trading en tiempo real"""
        print(f"\n{'='*80}")
        print("🔮 GENERANDO SEÑAL ACTUAL")
        print(f"{'='*80}")
        
        try:
            # Descargar datos recientes
            df_reciente = yf.download(
                self.ticker,
                start=self.fechas['actual'] - timedelta(days=7),
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df_reciente.empty:
                print("❌ No hay datos recientes")
                return
            
            # Procesar datos
            engineer = AdvancedFeatureEngineer()
            df_procesado = engineer.calcular_features_avanzadas(df_reciente)
            
            # Obtener predicciones de todos los horizontes
            predicciones = {}
            for horizonte, ensemble in self.modelos.items():
                pred = ensemble.predecir(df_procesado.tail(100))
                predicciones[horizonte] = pred
            
            if not predicciones:
                print("❌ No se pudieron generar predicciones")
                return
            
            # Analizar consenso
            señales = []
            confianzas = []
            
            for horizonte, pred in predicciones.items():
                if len(pred['prediccion']) > 0:
                    señales.append(pred['prediccion'][-1])
                    confianzas.append(pred['confianza'][-1])
            
            if not señales:
                print("❌ No hay señales disponibles")
                return
            
            # Votación ponderada
            señal_final = np.average(señales, weights=confianzas)
            confianza_promedio = np.mean(confianzas)
            consenso = np.std(señales)  # Baja desviación = alto consenso
            
            # Determinar dirección
            if señal_final > 0.5:
                dirección = "LONG"
                probabilidad = señal_final
            else:
                dirección = "SHORT"
                probabilidad = 1 - señal_final
            
            # Calcular niveles
            ultima_vela = df_procesado.iloc[-1]
            precio = ultima_vela['Close']
            atr = ultima_vela.get('ATR', precio * 0.02)
            
            if dirección == "LONG":
                sl = precio * (1 - TradingConfig.SL_MULTIPLIER * atr / precio)
                tp = precio * (1 + TradingConfig.TP_MULTIPLIER * atr / precio)
            else:
                sl = precio * (1 + TradingConfig.SL_MULTIPLIER * atr / precio)
                tp = precio * (1 - TradingConfig.TP_MULTIPLIER * atr / precio)
            
            ratio_rr = abs(tp - precio) / abs(precio - sl)
            
            # Mostrar señal
            print(f"\n📡 SEÑAL GENERADA:")
            print(f"  Dirección: {dirección}")
            print(f"  Probabilidad: {probabilidad:.2%}")
            print(f"  Confianza: {confianza_promedio:.2%}")
            print(f"  Consenso: {1-consenso:.2%}")
            print(f"  Precio: ${precio:,.2f}")
            print(f"  Stop Loss: ${sl:,.2f}")
            print(f"  Take Profit: ${tp:,.2f}")
            print(f"  Ratio R:R: {ratio_rr:.2f}")
            
            # Enviar por Telegram si cumple criterios
            if (confianza_promedio > TradingConfig.UMBRAL_CONFIANZA and 
                probabilidad > TradingConfig.UMBRAL_PROBABILIDAD and
                ratio_rr > TradingConfig.RATIO_RR_MINIMO):
                
                mensaje = (
                    f"🚨 SEÑAL {self.ticker}\n"
                    f"📅 {datetime.now(TradingConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M')}\n"
                    f"📈 Dirección: {dirección}\n"
                    f"🎯 Probabilidad: {probabilidad:.2%}\n"
                    f"🛡️ Confianza: {confianza_promedio:.2%}\n"
                    f"🤝 Consenso: {1-consenso:.2%}\n\n"
                    f"💰 Entrada: ${precio:,.2f}\n"
                    f"🛑 Stop Loss: ${sl:,.2f}\n"
                    f"🎯 Take Profit: ${tp:,.2f}\n"
                    f"⚖️ Ratio R:R: {ratio_rr:.2f}\n\n"
                    f"📊 RSI: {ultima_vela.get('RSI', 0):.1f}\n"
                    f"📈 Volatilidad: {ultima_vela.get('VOLATILITY_24H', 0)*100:.1f}%\n"
                    f"💹 ADX: {ultima_vela.get('ADX', 0):.1f}"
                )
                
                enviar_telegram(mensaje)
                
                # Guardar última señal
                guardar_ultima_senal({
                    "ticker": self.ticker,
                    "direccion": dirección,
                    "probabilidad": probabilidad,
                    "fecha": str(datetime.now(TradingConfig.TIMEZONE))
                })
            
        except Exception as e:
            print(f"❌ Error generando señal: {e}")


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

def main():
    print("🚀 SISTEMA DE TRADING AVANZADO CON OPTIMIZACIÓN")
    print("=" * 80)
    print("Incluye:")
    print("  • 50+ features técnicas (TA-Lib)")
    print("  • 5 algoritmos de ML con optimización de hiperparámetros")
    print("  • Ensembles avanzados")
    print("  • 4 estrategias de trading")
    print("  • 8 métricas de evaluación")
    print("  • Validación walk-forward rigurosa")
    print("=" * 80)
    
    # Verificar TA-Lib
    try:
        import talib
        print("✅ TA-Lib cargado correctamente")
    except ImportError:
        print("❌ TA-Lib no está instalado. Instala con: pip install TA-Lib")
        return
    
    resultados_totales = {}
    
    for ticker in TradingConfig.ACTIVOS:
        sistema = AdvancedTradingSystem(ticker)
        
        print(f"\n{'='*80}")
        print(f"🎯 PROCESANDO {ticker}")
        print(f"{'='*80}")
        
        if sistema.ejecutar_pipeline_completo():
            resultados_totales[ticker] = sistema.resultados
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN FINAL DEL SISTEMA")
    print(f"{'='*80}")
    
    viables = []
    for ticker, resultados in resultados_totales.items():
        if resultados:
            print(f"\n{ticker}:")
            print(f"  Operaciones: {resultados['n_operaciones']}")
            print(f"  Win Rate: {resultados['win_rate']:.2%}")
            print(f"  Retorno Total: {resultados['retorno_total']:.2%}")
            print(f"  Profit Factor: {resultados['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {resultados['sharpe_ratio']:.2f}")
            
            # Evaluar viabilidad simple
            criterios = (
                resultados['win_rate'] > 0.50,
                resultados['profit_factor'] > 1.3,
                resultados['n_operaciones'] >= 15,
                resultados['expectancy'] > 0
            )
            
            if sum(criterios) >= 3:
                viables.append(ticker)
                print(f"  ✅ VIABLE")
            else:
                print(f"  ⚠️ NO VIABLE")
    
    print(f"\n🎯 Tickers viables: {len(viables)}/{len(resultados_totales)}")
    if viables:
        print(f"   {', '.join(viables)}")
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
