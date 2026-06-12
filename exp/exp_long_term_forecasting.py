from data_provider.data_factory import *
from exp.exp_basic import *
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import calculate_metrics, calculate_financial_metrics
import torch
from datetime import datetime
import torch.nn as nn
from torch import optim
import os, time, warnings, gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.tools import align_predictions

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._debug_news_val_logged = False
        self._debug_news_test_logged = False
        self.log(f"\nExperiment begins at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    @property
    def log_file(self): return 'results.txt'
        
    def log(self, msg, verbose=True):
        if verbose: print(msg)
        with open(
                os.path.join(self.output_folder, self.log_file), 'a'
            ) as output_file:
            output_file.write(msg+'\n')

    @property
    def epoch_metrics_path(self):
        return os.path.join(self.output_folder, 'epoch_metrics.csv')

    def _init_epoch_metrics(self):
        pd.DataFrame(columns=[
            'epoch',
            'train_scaled_mse',
            'val_scaled_mse',
            'learning_rate',
            'epoch_time_sec',
        ]).to_csv(self.epoch_metrics_path, index=False)

    def _append_epoch_metrics(self, epoch, train_loss, val_loss, learning_rate, epoch_time):
        pd.DataFrame([{
            'epoch': int(epoch),
            'train_scaled_mse': float(train_loss),
            'val_scaled_mse': float(val_loss),
            'learning_rate': float(learning_rate),
            'epoch_time_sec': float(epoch_time),
        }]).to_csv(
            self.epoch_metrics_path,
            mode='a',
            header=not os.path.exists(self.epoch_metrics_path),
            index=False
        )

    def _prepare_batch(self, batch):
        """
        Backward-compatible batch unpacking.

        Original datasets yield:
        - seq_x, seq_y, seq_x_mark, seq_y_mark

        Phase-1 news-aware dataset additionally yields:
        - news_embeddings, news_time_gaps, news_mask, news_novelty, news_duration_probs
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]  #++
        extras = {}
        if len(batch) > 4:  #++
            extras['news_embeddings'] = batch[4].float().to(self.device)  #++
            extras['news_time_gaps'] = batch[5].float().to(self.device)  #++
            extras['news_mask'] = batch[6].float().to(self.device)  #++
            extras['news_novelty'] = batch[7].float().to(self.device)  #++
        if len(batch) > 8:  #++
            extras['news_duration_probs'] = batch[8].float().to(self.device)  #++
        if len(batch) > 9:  #++
            extras['news_sentiment'] = batch[9].float().to(self.device)  #++

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, extras

    def _run_model(self, batch_x, batch_y, batch_x_mark, batch_y_mark, extras):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.model.endswith('NewsDecay'):  #++
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **extras)  #++
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
        return outputs

    def _news_debug_source(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _format_debug_value(self, value):
        if value is None:
            return 'NA'
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        return f'{float(value):.4f}'

    def _format_news_debug_stats(self, stats):
        ordered_keys = [
            'use_market_state_decay',
            'disable_time_decay',
            'use_news_selector',
            'use_channel_specific_news',
            'use_novelty',
            'use_novelty_persistence',
            'use_duration_persistence',
            'active_news_mean',
            'time_gap_mean', 'time_gap_std', 'time_gap_min', 'time_gap_max',
            'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
            'selector_gate_mean', 'selector_gate_std', 'selector_gate_min', 'selector_gate_max',
            'novelty_mean', 'novelty_std', 'novelty_min', 'novelty_max',
            'persistence_bonus_mean', 'persistence_bonus_std', 'persistence_bonus_min', 'persistence_bonus_max',
            'gate_mean', 'gate_std', 'gate_min', 'gate_max',
            'weight_mean', 'weight_std', 'weight_min', 'weight_max',
            'duration_short_share', 'duration_long_share', 'duration_unsure_share',
        ]
        parts = []
        for key in ordered_keys:
            if key in stats:
                parts.append(f'{key}={self._format_debug_value(stats[key])}')
        return ', '.join(parts)

    def _maybe_log_news_debug(self, stage, batch_idx, epoch=None):
        if not getattr(self.args, 'debug_news_stats', False):
            return

        should_log = False
        if stage == 'train':
            every = max(1, int(getattr(self.args, 'debug_news_stats_every', 200)))
            should_log = batch_idx == 1 or (batch_idx % every) == 0
        elif getattr(self.args, 'debug_news_stats_in_val', False):
            if stage == 'val' and not self._debug_news_val_logged:
                self._debug_news_val_logged = True
                should_log = True
            elif stage == 'test' and not self._debug_news_test_logged:
                self._debug_news_test_logged = True
                should_log = True

        if not should_log:
            return

        stats = getattr(self._news_debug_source(), 'latest_debug_stats', None)
        if not stats:
            return

        prefix = f'[{stage}]'
        if epoch is not None:
            prefix += f' epoch={epoch}'
        prefix += f' batch={batch_idx}'
        self.log(f'{prefix} news_debug: {self._format_news_debug_stats(stats)}')

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()
    
    def _select_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, factor=0.1,
            verbose=True, min_lr=5e-6
        )

    def vali(self, vali_loader, criterion):
        total_loss = []
        
        self.model.eval()
        f_dim = -1 if self.args.features == 'MS' else 0
        
        progress_bar =tqdm(
            vali_loader, desc=f'Validation', 
            disable=self.args.disable_progress
        )
        self._debug_news_val_logged = False
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar, start=1):
                batch_x, batch_y, batch_x_mark, batch_y_mark, extras = self._prepare_batch(batch)
                outputs = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark, extras)
                self._maybe_log_news_debug(stage='val', batch_idx=batch_idx)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        
        self.model.train()
            
        return total_loss

    def train(self):
        if self.args.percent == 0:
            print('Zero shot learning, no need to train')
            return
        
        _, train_loader = self.get_data(flag='train')
        _, vali_loader = self.get_data(flag='val')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            self.output_folder, 
            patience=self.args.patience, verbose=True,
            best_model_name=self.best_model_name
        )

        model_optim = self._select_optimizer()
            
        criterion = self._select_criterion()
        lr_scheduler = self._select_lr_scheduler(model_optim)
        self._init_epoch_metrics()
        
        f_dim = -1 if self.args.features == 'MS' else 0
        
        
        for epoch in range(self.args.train_epochs):
            progress_bar =tqdm(
                train_loader, desc=f'Training: Epoch {epoch+1}: ', 
                disable=self.args.disable_progress
            )

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, batch in enumerate(progress_bar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark, extras = self._prepare_batch(batch)
                batch_y = batch_y.to(self.device)
                outputs = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark, extras)
                self._maybe_log_news_debug(stage='train', batch_idx=i + 1, epoch=epoch + 1)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print(f"\titers: {i + 1} | loss: {loss.item():.5g}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4g}s/iter; left time: {left_time:.4g}s')
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                model_optim.step()
            
            train_loss = np.average(train_loss)
            
            val_loss = self.vali(vali_loader, criterion)

            epoch_elapsed = time.time() - epoch_time
            current_lr = model_optim.param_groups[0]['lr']
            self._append_epoch_metrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_time=epoch_elapsed,
            )
            print(f"Epoch: {epoch + 1} | Time: {epoch_elapsed:0.3g} s | Train Loss: {train_loss:.5g} Vali Loss: {val_loss:.5g}")
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr_scheduler.step(val_loss)
            gc.collect()
        
        time_per_epoch = (time.time() - time_now) / (epoch + 1)
        print(f"\nTraining completed at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.")
        gc.collect()
            
        self.profile(time_per_epoch)
        self.load_best_model()
        return self.model
    
    def profile(self, time_per_epoch):
        # add if p.requires_grad to count only trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters()) 
        self.log(f"Model parameters: {total_params}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        # Get the current memory allocated by PyTorch on the GPU
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**2
        # Get the maximum memory allocated by PyTorch on the GPU
        max_allocated_memory = torch.cuda.max_memory_allocated(0) / 1024**2

        print(f"Total memory: {total_memory:.1f} MB")
        print(f"Allocated memory: {allocated_memory:.1f} MB")
        print(f"Max allocated memory: {max_allocated_memory:.1f} MB")
        # print(torch.cuda.memory_summary())
        
        self.log(f"Time per epoch: {time_per_epoch:.1f} sec.")
        self.log(f"Memory usage: Available {total_memory:.1f} MB, Allocated {allocated_memory:.1f} MB, Max allocated {max_allocated_memory:.1f} MB\n")
    
    def test(
        self, load_model:bool=True, flag='test', 
        evaluate=True, dump_output=False, 
        remove_negative=True
    ):
        test_data, test_loader = self.get_data(flag)
        
        # percent 0 is for zero-shot learning, no need to load model
        if (load_model or self.args.test) and self.args.percent > 0:
            self.load_best_model()
        else:
            print('No need to load model')
            
        disable_progress = self.args.disable_progress

        preds = []
        trues = []
        inputs = []
        f_dim = -1 if self.args.features == 'MS' else 0

        self.model.eval()
        self._debug_news_test_logged = False
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(test_loader), desc="Running inference",
                total=len(test_loader), disable=disable_progress
            ):
                batch_x, batch_y, batch_x_mark, batch_y_mark, extras = self._prepare_batch(batch)
                batch_y = batch_y.to(self.device)
                outputs = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark, extras)
                self._maybe_log_news_debug(stage='test', batch_idx=i + 1)

                outputs = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                inputs.append(batch_x_np)

        # this line handles different size of batch. E.g. last batch can be < batch_size.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        data_name = self.args.data_path.split('.')[0]

        print('Preds and Trues shape:', preds.shape, trues.shape)
        if evaluate:
            # calculate evaluations
            mae, rmse, _, _ = calculate_metrics(preds, trues)
            mse = rmse ** 2
            
            # dump results in the global file
            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(data_name + " " + self.setting + "  " + flag + " scaled\n")
                f.write(f'mae:{mae:.5g}, rmse:{rmse:.5g}, mse:{mse:.5g}\n\n')

            # dump results in the respective result folder
            self.log(f'{flag} scaled -- mse:{mse:.5g}, mae:{mae:.5g}')
        
        # inverse transform and remove negatives
        print("Upscaling data and removing negatives...")
    
        for i in range(preds.shape[0]):
            # date = test_data.index.loc[i, 'date']
            scaler = test_data.scaler[i]
            preds[i] = test_data.inverse_transform(scaler, preds[i])
            trues[i] = test_data.inverse_transform(scaler, trues[i])
            inputs[i] = test_data.inverse_transform(scaler, inputs[i])
    
        if remove_negative:
            print("Removing negatives...")
            preds[preds<0] = 0
        # print('Trues ', trues)
        # print('Preds ', preds)
        
        if evaluate:
            # calculate evaluations
            mae, rmse, rmsle, smape = calculate_metrics(preds, trues)
            mse = rmse ** 2

            close_idx = 0
            close_candidates = ['Close/Last', 'Close', 'close']
            if isinstance(test_data.target, list):
                for name in close_candidates:
                    if name in test_data.target:
                        close_idx = test_data.target.index(name)
                        break
            fin_metrics = calculate_financial_metrics(
                last_close=inputs[:, -1, close_idx],
                pred_close=preds[:, 0, close_idx],
                true_close=trues[:, 0, close_idx],
            )
            
            # dump results in the global file
            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(data_name + " " + self.setting + "  " + flag + "\n")
                f.write(f'mae:{mae:.5g}, rmse:{rmse:.5g}, mse:{mse:.5g}, rmsle {rmsle:0.5g} smape {smape:0.5g}\n\n')
                f.write(
                    "financial "
                    + " ".join([f"{k}:{v:.5g}" for k, v in fin_metrics.items()])
                    + "\n\n"
                )

            # dump results in the respective result folder
            self.log(f'{flag} -- mse:{mse:.5g}, mae:{mae:.5g}, rmsle: {rmsle:0.5g} smape {smape:0.5g}\n')
            self.log(
                f"{flag} financial -- "
                + ", ".join([f"{k}:{v:.5g}" for k, v in fin_metrics.items()])
                + "\n"
            )
                
        if dump_output:
            # get ground truth
            target, time_col = test_data.target, test_data.time_col
            selected_columns = [time_col]
            
            if type(test_data) == MultiTimeSeries: 
                selected_columns.append(test_data.group_id)
            
            if type(target) == list: selected_columns += target
            else: selected_columns.append(target)
            
            filepath = os.path.join(self.args.root_path, self.args.data_path)
            ground_truth = pd.read_csv(filepath)[selected_columns]
            
            if ground_truth[test_data.time_col].dtype == 'object':
                ground_truth[test_data.time_col] = pd.to_datetime(ground_truth[test_data.time_col])
                
            # output prediction into a csv file
            merged = align_predictions(
                ground_truth, preds, test_data, 
                remove_negative=False, upscale=False, 
                disable_progress=self.args.disable_progress
            )
            merged.round(4).to_csv(
                os.path.join(self.output_folder, f'{flag}.csv'), 
                index=False
            )
            
        gc.collect()
