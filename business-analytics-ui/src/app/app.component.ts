import { Component, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer } from '@angular/platform-browser';
import { NewlinePipe } from './newline.pipe';
import Plotly from 'plotly.js-dist';
import { FileManagementComponent } from './file-management/file-management.component';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { environment } from '../environments/environment';

interface Message {
  content: string;
  isUser: boolean;
  graph?: {
    data: any[];
    layout: any;
  };
}

interface UploadedFile {
  name: string;
  path: string;
  type: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    NewlinePipe, 
    FileManagementComponent,
    MatCardModule,
    MatIconModule,
    MatButtonModule
  ]
})
export class AppComponent implements AfterViewChecked {
  @ViewChild('chatContainer') private chatContainer!: ElementRef;
  
  messages: Message[] = [];
  userInput: string = '';
  uploadedFiles: UploadedFile[] = [];
  isInputDisabled: boolean = true;
  
  constructor(
    private http: HttpClient,
    private sanitizer: DomSanitizer
  ) {
    // Add welcome message
    this.messages.push({
      content: 'Hello! I\'m your Business Analysis Assistant. Please upload your files and I\'ll help you analyze them.',
      isUser: false
    });
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  public scrollToBottom(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  scrollToTop(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = 0;
    } catch (err) {
      console.error('Error scrolling to top:', err);
    }
  }

  onFilesChanged(files: UploadedFile[]): void {
    console.log('Files changed:', files);
    this.uploadedFiles = files;
    this.isInputDisabled = files.length === 0;
    
    if (files.length > 0 && this.messages.length === 1) {
      this.messages.push({
        content: 'Great! Your files have been uploaded. What would you like to know about them?',
        isUser: false
      });
    }
  }

  async sendMessage() {
    if (!this.userInput.trim() || this.isInputDisabled) return;

    const userQuestion = this.userInput.trim();
    
    // Add user message
    this.messages.push({
      content: userQuestion,
      isUser: true
    });

    this.userInput = '';
    this.scrollToBottom();

    // Show loading indicator
    const loadingMessage: Message = {
      content: 'Analyzing your request...',
      isUser: false
    };
    this.messages.push(loadingMessage);

    try {
      const isGraphQuestion = 
        userQuestion.toLowerCase().includes('show') || 
        userQuestion.toLowerCase().includes('display') ||
        userQuestion.toLowerCase().includes('graph') ||
        userQuestion.toLowerCase().includes('chart') ||
        userQuestion.toLowerCase().includes('plot') ||
        userQuestion.toLowerCase().includes('visualize');

      if (isGraphQuestion && this.uploadedFiles.length > 0) {
        loadingMessage.content = "I'll generate a visualization based on your request.";
        
        try {
          const response = await this.http.post<any>(
            `${environment.apiUrl}/generate_graph`,
            {
              question: userQuestion,
              file_path: this.uploadedFiles[0].path
            }
          ).toPromise();

          if (response?.graph) {
            loadingMessage.graph = response.graph;
            
            setTimeout(() => {
              const lastIndex = this.messages.findIndex(msg => msg === loadingMessage);
              if (lastIndex !== -1) {
                const graphContainer = document.getElementById(`graph-${lastIndex}`);
                if (graphContainer) {
                  Plotly.newPlot(
                    graphContainer, 
                    response.graph.data, 
                    {
                      ...response.graph.layout,
                      autosize: true,
                      responsive: true
                    },
                    {
                      responsive: true,
                      displayModeBar: true
                    }
                  );
                }
              }
            }, 100);
          }
        } catch (error) {
          console.error('Graph generation error:', error);
          loadingMessage.content = 'Sorry, I encountered an error generating the graph. Please try a different question.';
        }
      } else {
        try {
          const response = await this.http.post<any>(
            `${environment.apiUrl}/analyze`,
            {
              question: userQuestion,
              file_paths: this.uploadedFiles.map(f => f.path)
            }
          ).toPromise();

          loadingMessage.content = response?.message || "I couldn't generate a response for your question.";
        } catch (error) {
          console.error('Analysis error:', error);
          loadingMessage.content = 'Sorry, I encountered an error analyzing your request. Please try again.';
        }
      }
    } catch (error) {
      console.error('Error:', error);
      loadingMessage.content = 'Sorry, I encountered an error. Please try again.';
    }
  }
}
