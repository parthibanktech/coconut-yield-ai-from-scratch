import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { PredictionResponse } from './models/prediction.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  template: `
    <div class="container">
      <header>
        <h1>🥥 Coconut Yield Predictor</h1>
        <p>AI-Powered Agriculture</p>
      </header>
      
      <main>
        <div class="card input-section">
          <h2>Enter Farm Details</h2>
          <div class="form-group">
            <label for="area">Farm Area (Hectares)</label>
            <input type="number" id="area" [(ngModel)]="area" placeholder="e.g. 1000">
          </div>
          <button (click)="predict()" [disabled]="!area || loading">
            {{ loading ? 'Calculating...' : 'Predict Yield' }}
          </button>
        </div>

        <div *ngIf="result" class="card result-section">
          <h2>Prediction Result</h2>
          <div class="result-value">
            <span class="value">{{ result.predicted_production | number:'1.0-0' }}</span>
            <span class="unit">Coconuts</span>
          </div>
          <p class="description">
            Estimated annual production for a {{ result.area }} hectare farm.
          </p>
        </div>

        <div *ngIf="error" class="error-message">
          {{ error }}
        </div>
      </main>
    </div>
  `,
  styles: [`
    :host {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      display: block;
      height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      color: #333;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      padding: 2rem;
      text-align: center;
    }
    header h1 {
      color: #2c3e50;
      margin-bottom: 0.5rem;
    }
    header p {
      color: #7f8c8d;
      margin-bottom: 2rem;
    }
    .card {
      background: white;
      border-radius: 15px;
      padding: 2rem;
      box-shadow: 0 10px 20px rgba(0,0,0,0.1);
      margin-bottom: 2rem;
      transition: transform 0.3s ease;
    }
    .card:hover {
      transform: translateY(-5px);
    }
    .form-group {
      margin-bottom: 1.5rem;
      text-align: left;
    }
    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 600;
      color: #34495e;
    }
    input {
      width: 100%;
      padding: 0.8rem;
      border: 2px solid #ecf0f1;
      border-radius: 8px;
      font-size: 1rem;
      transition: border-color 0.3s;
    }
    input:focus {
      border-color: #3498db;
      outline: none;
    }
    button {
      background: #2ecc71;
      color: white;
      border: none;
      padding: 1rem 2rem;
      font-size: 1.1rem;
      border-radius: 50px;
      cursor: pointer;
      transition: background 0.3s;
      width: 100%;
    }
    button:hover {
      background: #27ae60;
    }
    button:disabled {
      background: #bdc3c7;
      cursor: not-allowed;
    }
    .result-value {
      font-size: 3rem;
      font-weight: bold;
      color: #2c3e50;
      margin: 1rem 0;
    }
    .unit {
      font-size: 1.2rem;
      color: #7f8c8d;
      margin-left: 0.5rem;
    }
    .error-message {
      color: #e74c3c;
      background: #fadbd8;
      padding: 1rem;
      border-radius: 8px;
      margin-top: 1rem;
    }
  `]
})
export class App {
  area: number | null = null;
  result: PredictionResponse | null = null;
  loading = false;
  error: string | null = null;

  constructor(private http: HttpClient) { }

  predict() {
    if (!this.area) return;

    this.loading = true;
    this.error = null;
    this.result = null;

    this.http.post<PredictionResponse>('http://127.0.0.1:5000/predict', { area: this.area })
      .subscribe({
        next: (data) => {
          this.result = data;
          this.loading = false;
        },
        error: (err) => {
          console.error('Error:', err);
          this.error = 'Failed to get prediction. Please try again.';
          this.loading = false;
        }
      });
  }
}
