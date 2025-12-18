import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  activeTab: 'predict' | 'data' = 'predict';

  // Prediction State
  area: number | null = null;
  year: number | null = null;
  result: any | null = null;
  loading = false;
  error: string | null = null;

  // Data Science State
  dataInsight: any | null = null;
  dataLoading = false;
  dataError: string | null = null;
  private apiUrl = 'http://127.0.0.1:8080';

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef, private sanitizer: DomSanitizer) {
    this.loadData();
  }

  selectTab(tab: 'predict' | 'data') {
    this.activeTab = tab;
    if (tab === 'data') {
      this.loadData();
    }
  }


  predict() {
    if (!this.area) return;

    this.loading = true;
    this.error = null;
    this.result = null;

    const payload = {
      area: this.area,
      year: this.year ? this.year : ""
    };

    console.log('Sending payload:', payload);

    this.http.post<any>(`${this.apiUrl}/predict`, payload)
      .subscribe({
        next: (data) => {
          console.log('Received response:', data);
          if (data.success) {
            // Sanitize images
            if (data.context_plot) {
              data.context_plot = this.sanitizer.bypassSecurityTrustResourceUrl(data.context_plot);
            }
            if (data.global_context_plot) {
              data.global_context_plot = this.sanitizer.bypassSecurityTrustResourceUrl(data.global_context_plot);
            }
            this.result = data;
          } else {
            this.error = data.error || 'Unknown error from server';
          }
          this.loading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('Error:', err);
          this.error = 'Failed to connect to the AI model. Ensure the backend is running.';
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }

  loadData() {
    if (this.dataInsight) return;

    this.dataLoading = true;
    this.dataError = null;

    this.http.get<any>(`${this.apiUrl}/data`)
      .subscribe({
        next: (data) => {
          // Sanitize images
          if (data.plot_image) {
            data.plot_image = this.sanitizer.bypassSecurityTrustResourceUrl(data.plot_image);
          }
          if (data.cost_plot) {
            data.cost_plot = this.sanitizer.bypassSecurityTrustResourceUrl(data.cost_plot);
          }
          if (data.residual_plot) {
            data.residual_plot = this.sanitizer.bypassSecurityTrustResourceUrl(data.residual_plot);
          }
          this.dataInsight = data;
          this.dataLoading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('Data Load Error:', err);
          this.dataError = 'Failed to load data. Is the backend server running?';
          this.dataLoading = false;
          this.cdr.detectChanges();
        }
      });
  }
}
