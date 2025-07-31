import { Component, OnInit, OnDestroy, Output, EventEmitter } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { MatSnackBar } from '@angular/material/snack-bar';
import { environment } from '../../environments/environment';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { FileSizePipe } from '../pipes/file-size.pipe';

export interface UploadedFile {
  name: string;
  path: string;
  type: string;
}

@Component({
  selector: 'app-file-management',
  templateUrl: './file-management.component.html',
  styleUrls: ['./file-management.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatListModule,
    MatProgressBarModule,
    MatButtonModule,
    FileSizePipe
  ]
})
export class FileManagementComponent implements OnInit, OnDestroy {
  @Output() filesChanged = new EventEmitter<UploadedFile[]>();
  
  files: File[] = [];
  existingFiles: UploadedFile[] = [];
  uploadProgress: number = 0;
  isUploading: boolean = false;
  dragover: boolean = false;

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadExistingFiles();
  }

  ngOnDestroy(): void {}

  onFileSelected(event: any): void {
    const files = event.target?.files;
    if (files && files.length > 0) {
      this.files = Array.from(files);
      this.uploadFiles();
    }
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragover = false;
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.files = Array.from(files);
      this.uploadFiles();
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragover = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragover = false;
  }

  async uploadFiles(): Promise<void> {
    if (!this.files.length) return;

    this.isUploading = true;
    const formData = new FormData();
    this.files.forEach(file => formData.append('files', file));

    try {
      const response = await this.http.post<UploadedFile[]>(`${environment.apiUrl}/upload`, formData).toPromise();
      if (response) {
        this.existingFiles = [...this.existingFiles, ...response];
        this.filesChanged.emit(this.existingFiles);
        this.showMessage('Files uploaded successfully');
      }
    } catch (error) {
      this.handleError(error);
    } finally {
      this.isUploading = false;
      this.files = [];
      this.uploadProgress = 0;
    }
  }

  async deleteFile(file: UploadedFile): Promise<void> {
    try {
      await this.http.delete(`${environment.apiUrl}/files/${encodeURIComponent(file.path)}`).toPromise();
      this.existingFiles = this.existingFiles.filter(f => f.path !== file.path);
      this.filesChanged.emit(this.existingFiles);
      this.showMessage(`File ${file.name} deleted successfully`);
    } catch (error) {
      this.handleError(error);
    }
  }

  private async loadExistingFiles(): Promise<void> {
    try {
      const files = await this.http.get<UploadedFile[]>(`${environment.apiUrl}/files`).toPromise();
      if (files) {
        this.existingFiles = files;
        this.filesChanged.emit(this.existingFiles);
      }
    } catch (error) {
      this.handleError(error);
    }
  }

  private handleError(error: any): void {
    console.error('Error:', error);
    const message = error instanceof HttpErrorResponse && error.error?.message
      ? error.error.message
      : 'An error occurred';
    this.showMessage(message, true);
  }

  private showMessage(message: string, isError: boolean = false): void {
    this.snackBar.open(message, 'Close', {
      duration: 3000,
      panelClass: isError ? ['error-snackbar'] : ['success-snackbar']
    });
  }
} 