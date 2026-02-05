"""
Trading Cuantitativo - Sistema Optimizado sin TA-Lib
Versión: 3.2 - Compatible con GitHub Actions
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class Config:
    """Configuración optimizada"""
    # Assets principales de crypto
    SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
    
    # Timeframes
    INTERVAL = "1h"
    TRAIN_DAYS = 90  # 3 meses (más rápido)
    TEST_DAYS = 30   # 1 mes
    
    # Trading parameters
    COMMISSION = 0.001
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    
    # Model parameters
    MIN_DATA_POINTS = 50
    CONFIDENCE_THRESHOLD = 0.55

class TechnicalIndicators:
    """Cálculo manual de indicadores técnicos"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI manualmente"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Evitar división por cero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD manualmente"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2):
        """Calcula Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr

class DataHandler:
    """Manejo robusto de datos"""
    
    @staticmethod
    def get_data(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Obtiene datos con validación"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Descargando {symbol} desde {start_date.date()}")
            
            # Usar período en lugar de fechas exactas para mejor compatibilidad
            df = yf.download(
                symbol,
                period=f"{days}d",
                interval=Config.INTERVAL,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            
            if df.empty or len(df) < Config.MIN_DATA_POINTS:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # Renombrar y limpiar columnas
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Limpiar NaN
            df = df.dropna()
            
            logger.info(f"✅ {symbol}: {len(df)} velas, Precio: ${df['close'].iloc[-1]:.2f}")
            return df
            
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {str(e)[:100]}")
            return None

class FeatureEngineer:
    """Ingeniería de características sin dependencias externas"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Crea features técnicas manualmente"""
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. RETORNOS
        df['returns_1h'] = close.pct_change(1)
        df['returns_4h'] = close.pct_change(4)
        df['returns_12h'] = close.pct_change(12)
        
        # 2. MEDIAS MÓVILES
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        # 3. RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(close, 14)
        
        # 4. MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(close)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        # 5. BOLLINGER BANDS
        bb_middle, bb_upper, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, 1)
        
        # 6. ATR
        df['atr'] = TechnicalIndicators.calculate_atr(high, low, close)
        df['atr_pct'] = df['atr'] / close
        
        # 7. VOLUMEN
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1)
        
        # 8. TENDENCIA
        df['trend'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # 9. MOMENTUM
        df['momentum_12h'] = close / close.shift(12) - 1
        
        # 10. RANGOS
        df['high_24h'] = high.rolling(24).max()
        df['low_24h'] = low.rolling(24).min()
        df['range_position'] = (close - df['low_24h']) / (df['high_24h'] - df['low_24h']).replace(0, 1)
        
        # 11. VOLATILIDAD
        df['volatility_12h'] = df['returns_1h'].rolling(12).std()
        
        # 12. PRICE ACTION
        df['candle_body'] = (close - df['open']).abs() / close
        df['high_low_ratio'] = (high - low) / close
        
        # Eliminar NaN
        df = df.dropna()
        
        logger.info(f"Features: {len(df.columns)} columnas, {len(df)} filas válidas")
        return df

class LabelGenerator:
    """Generación de etiquetas simplificada"""
    
    @staticmethod
    def create_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
        """Crea etiquetas binarias simples"""
        # Precio futuro
        future_price = df['close'].shift(-horizon)
        
        # Retorno futuro
        future_return = (future_price / df['close'] - 1)
        
        # Umbral fijo del 1%
        threshold = 0.01
        
        # Etiquetas
        labels = pd.Series(0, index=df.index)  # Por defecto 0 (neutral/sell)
        labels[future_return > threshold] = 1  # 1 = comprar
        
        # Filtrar solo donde tenemos datos futuros
        labels = labels[future_return.notna()]
        
        # Estadísticas
        n_buy = (labels == 1).sum()
        n_total = len(labels)
        
        if n_total > 0:
            logger.info(f"Etiquetas: BUY={n_buy} ({n_buy/n_total:.1%}), SELL/NEUTRAL={n_total-n_buy}")
        
        return labels

class TradingModel:
    """Modelo de ML simplificado"""
    
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        """Entrena modelo XGBoost"""
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Seleccionar features manualmente (las más importantes)
            feature_candidates = [
                'returns_1h', 'returns_4h', 'returns_12h',
                'rsi', 'macd', 'macd_hist', 'bb_position',
                'atr_pct', 'volume_ratio', 'trend',
                'momentum_12h', 'range_position', 'volatility_12h'
            ]
            
            # Filtrar features disponibles
            self.features = [f for f in feature_candidates if f in X_train.columns]
            
            if len(self.features) < 5:
                logger.warning("Features insuficientes")
                return False
            
            # Preparar datos
            X = X_train[self.features].fillna(0)
            y = y_train.values
            
            # Escalar
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar modelo simple
            self.model = XGBClassifier(
                n_estimators=50,  # Reducido para velocidad
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            self.model.fit(X_scaled, y)
            
            # Validación simple
            train_pred = self.model.predict(X_scaled)
            accuracy = (train_pred == y).mean()
            
            logger.info(f"Modelo entrenado - Accuracy: {accuracy:.2%}, Features: {len(self.features)}")
            return accuracy > 0.5
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicciones"""
        if self.model is None or self.features is None:
            return np.zeros(len(X)), np.zeros(len(X))
        
        try:
            # Preparar datos
            available_features = [f for f in self.features if f in X.columns]
            X_data = X[available_features].fillna(0)
            
            # Si faltan features, añadir ceros
            if len(available_features) < len(self.features):
                missing = set(self.features) - set(available_features)
                for f in missing:
                    X_data[f] = 0
            
            X_scaled = self.scaler.transform(X_data[self.features])
            
            # Predecir
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.zeros(len(X)), np.zeros(len(X))

class SignalGenerator:
    """Genera señales de trading"""
    
    @staticmethod
    def generate(df: pd.DataFrame, predictions: np.ndarray, 
                 probabilities: np.ndarray, threshold: float = 0.55) -> pd.DataFrame:
        """Genera señales con filtros básicos"""
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = predictions
        signals['probability'] = probabilities
        
        # Señal base
        signals['signal'] = 0
        buy_signals = (signals['prediction'] == 1) & (signals['probability'] > threshold)
        signals.loc[buy_signals, 'signal'] = 1
        
        # Filtrar RSI extremos
        if 'rsi' in df.columns:
            overbought = df['rsi'] > 70
            oversold = df['rsi'] < 30
            signals.loc[(signals['signal'] == 1) & overbought, 'signal'] = 0
        
        # Estadísticas
        n_signals = (signals['signal'] != 0).sum()
        logger.info(f"Señales generadas: {n_signals}")
        
        return signals[['signal', 'probability']]

class Backtester:
    """Backtesting simple"""
    
    @staticmethod
    def run(df: pd.DataFrame, signals: pd.DataFrame, 
            commission: float = 0.001, 
            stop_loss: float = 0.02,
            take_profit: float = 0.04) -> Dict:
        """Ejecuta backtesting básico"""
        try:
            capital = 10000
            position = 0
            trades = []
            
            for i in range(1, len(df)):
                price = df['close'].iloc[i]
                signal = signals['signal'].iloc[i]
                
                # Si tenemos posición, verificar stop/take
                if position != 0:
                    entry_price = trades[-1]['entry_price']
                    if position > 0:  # Long
                        pnl_pct = (price - entry_price) / entry_price
                        if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                            # Cerrar posición
                            capital += position * price * (1 - commission)
                            trades[-1].update({
                                'exit_price': price,
                                'pnl_pct': pnl_pct,
                                'exit_reason': 'SL' if pnl_pct <= -stop_loss else 'TP'
                            })
                            position = 0
                    # Para short sería similar
                
                # Abrir nueva posición si hay señal
                if position == 0 and signal == 1:
                    position_size = capital * 0.1 / price  # 10% del capital
                    capital -= position_size * price * (1 + commission)
                    position = position_size
                    
                    trades.append({
                        'entry_price': price,
                        'position_size': position_size,
                        'entry_time': df.index[i]
                    })
            
            # Cerrar posición final si queda abierta
            if position != 0 and trades:
                last_price = df['close'].iloc[-1]
                entry_price = trades[-1]['entry_price']
                pnl_pct = (last_price - entry_price) / entry_price
                capital += position * last_price * (1 - commission)
                
                trades[-1].update({
                    'exit_price': last_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'END'
                })
            
            # Calcular métricas
            initial_capital = 10000
            final_capital = capital
            total_return = (final_capital / initial_capital - 1) * 100
            
            if trades:
                winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t.get('pnl_pct', 0) for t in winning_trades]) * 100 if winning_trades else 0
                
                results = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'win_rate': win_rate,
                    'avg_win_pct': avg_win,
                    'total_return_pct': total_return,
                    'final_capital': final_capital
                }
            else:
                results = {
                    'total_trades': 0,
                    'total_return_pct': 0,
                    'final_capital': initial_capital
                }
            
            return {
                'metrics': results,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error en backtesting: {e}")
            return None

class TradingSystem:
    """Sistema principal"""
    
    def __init__(self):
        self.config = Config()
        
    def process_symbol(self, symbol: str) -> Optional[Dict]:
        """Procesa un símbolo completo"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESANDO: {symbol}")
            logger.info(f"{'='*60}")
            
            # 1. Obtener datos
            df = DataHandler.get_data(symbol, self.config.TRAIN_DAYS + self.config.TEST_DAYS)
            if df is None or len(df) < 100:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # 2. Crear features
            df_features = FeatureEngineer.create_features(df)
            if len(df_features) < 50:
                logger.warning(f"Features insuficientes después de limpieza")
                return None
            
            # 3. Split train/test (80/20)
            split_idx = int(len(df_features) * 0.8)
            train_data = df_features.iloc[:split_idx].copy()
            test_data = df_features.iloc[split_idx:].copy()
            
            logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")
            
            # 4. Generar labels
            labels = LabelGenerator.create_labels(train_data)
            
            # 5. Entrenar modelo
            model = TradingModel()
            if not model.train(train_data, labels):
                logger.warning(f"Modelo no pudo ser entrenado para {symbol}")
                return None
            
            # 6. Predecir
            predictions, probabilities = model.predict(test_data)
            
            # 7. Generar señales
            signals = SignalGenerator.generate(test_data, predictions, probabilities, 
                                              self.config.CONFIDENCE_THRESHOLD)
            
            # 8. Backtest
            backtest_results = Backtester.run(
                test_data, signals, 
                self.config.COMMISSION,
                self.config.STOP_LOSS_PCT,
                self.config.TAKE_PROFIT_PCT
            )
            
            if backtest_results is None:
                return None
            
            # 9. Preparar resultados
            metrics = backtest_results['metrics']
            results = {
                'symbol': symbol,
                'data_points': len(df),
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'signals': len(signals[signals['signal'] != 0]),
                'trades': metrics['total_trades'],
                'win_rate': metrics.get('win_rate', 0),
                'total_return': metrics['total_return_pct'],
                'final_capital': metrics['final_capital']
            }
            
            # Mostrar resultados
            logger.info(f"\n📊 RESULTADOS {symbol}:")
            logger.info(f"  Operaciones: {results['trades']}")
            if results['trades'] > 0:
                logger.info(f"  Win Rate: {results['win_rate']:.1%}")
            logger.info(f"  Retorno Total: {results['total_return']:.2f}%")
            logger.info(f"  Capital Final: ${results['final_capital']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando {symbol}: {e}")
            return None

def main():
    """Función principal"""
    logger.info("🚀 SISTEMA DE TRADING CUANTITATIVO v3.2")
    logger.info("=" * 60)
    
    # Crear sistema
    system = TradingSystem()
    
    # Procesar símbolos
    all_results = {}
    
    for symbol in system.config.SYMBOLS:
        try:
            result = system.process_symbol(symbol)
            if result:
                all_results[symbol] = result
        except Exception as e:
            logger.error(f"Error con {symbol}: {e}")
            continue
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📈 RESUMEN FINAL")
    logger.info("=" * 60)
    
    if not all_results:
        logger.warning("No se obtuvieron resultados válidos")
        return None
    
    # Calcular estadísticas
    total_trades = sum(r['trades'] for r in all_results.values())
    avg_return = np.mean([r['total_return'] for r in all_results.values()])
    
    logger.info(f"Símbolos procesados: {len(all_results)}")
    logger.info(f"Operaciones totales: {total_trades}")
    logger.info(f"Retorno promedio: {avg_return:.2f}%")
    
    # Mejor y peor
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]['total_return'])
        worst = min(all_results.items(), key=lambda x: x[1]['total_return'])
        
        logger.info(f"\n🏆 MEJOR: {best[0]} - {best[1]['total_return']:.2f}%")
        logger.info(f"📉 PEOR: {worst[0]} - {worst[1]['total_return']:.2f}%")
    
    # Guardar resultados
    output_file = 'trading_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Resultados guardados en {output_file}")
    logger.info("\n✅ PROCESO COMPLETADO")
    
    return all_results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Proceso interrumpido")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {e}")
        sys.exit(1)
