import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import pickle
import datetime
import csv
from attendance_db import AttendanceDB
import threading
import queue
import os

class IntegratedAttendanceSystem:
    def __init__(self):
        self.db = AttendanceDB()
        self.is_running = False
        self.video_capture = None
        self.recognition_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Increased queue size
        self.setup_gui()
        self.load_encodings()
        self.current_image = None  # Store reference to prevent garbage collection

    def setup_gui(self):
        # GUI START
        self.root = tk.Tk()
        self.root.title("Integrated Attendance System")
        self.root.geometry("1200x800")
        
        # main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_container, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_recognition)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_recognition, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.export_button = ttk.Button(control_frame, text="Export to CSV", command=self.export_to_csv)
        self.export_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Status 
        self.status_label = ttk.Label(control_frame, text="System Ready")
        self.status_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Create left and right frames
        left_frame = ttk.Frame(main_container)
        right_frame = ttk.Frame(main_container)
        left_frame.grid(row=1, column=0, padx=5)
        right_frame.grid(row=1, column=1, padx=5)
        
        # Attendance Log
        log_frame = ttk.LabelFrame(left_frame, text="Today's Attendance Log", padding="5")
        log_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        #  Treeview for attendance 
        self.tree = ttk.Treeview(log_frame, columns=("Name", "Time", "Status"), show="headings", height=20)
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Time")
        self.tree.heading("Status", text="Status")
        
        #scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Stati
        stats_frame = ttk.LabelFrame(right_frame, text="Detailed Statistics", padding="5")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Detailed stat
        self.total_students_label = ttk.Label(stats_frame, text="Total Students: 0")
        self.total_students_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.present_students_label = ttk.Label(stats_frame, text="Present: 0")
        self.present_students_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.absent_students_label = ttk.Label(stats_frame, text="Absent: 0")
        self.absent_students_label.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.attendance_rate_label = ttk.Label(stats_frame, text="Attendance Rate: 0%")
        self.attendance_rate_label.grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Absent  
        absent_list_frame = ttk.LabelFrame(right_frame, text="Absent Students", padding="5")
        absent_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.absent_list = tk.Listbox(absent_list_frame, height=10)
        self.absent_list.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)

    def load_encodings(self):
        """Load the face encodings from the pickle file"""
        try:
            with open("encodings.pickle", "rb") as f:
                self.data = pickle.load(f)
            if "encodings" not in self.data or "names" not in self.data:
                raise ValueError("Missing required keys in encodings file")
            if len(self.data["encodings"]) == 0:
                raise ValueError("No encodings found")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading encodings: {str(e)}")
            self.root.quit()

    def start_recognition(self):
        """Start the face recognition process"""
        try:
            if not self.is_running:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    messagebox.showerror("Error", "Unable to open camera")
                    return
                
                # Test camera read
                ret, _ = self.video_capture.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to read from camera")
                    self.video_capture.release()
                    return
                
                self.is_running = True
                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.status_label.config(text="Recognition Running")
                
                # Start recognition thread
                self.recognition_thread = threading.Thread(target=self.recognition_loop)
                self.recognition_thread.daemon = True
                self.recognition_thread.start()
                
                # Start frame update
                self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition: {str(e)}")
            self.stop_recognition()

    def stop_recognition(self):
        """Stop the face recognition process"""
        try:
            self.is_running = False
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()
            
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Recognition Stopped")
            
            # Update statistics one final time
            self.update_statistics()
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping recognition: {str(e)}")

    def recognition_loop(self):
        """Main recognition loop running in separate thread"""
        frame_count = 0
        while self.is_running:
            ret, frame = self.video_capture.read()
            if not ret:
                continue
                
            # Process every 3rd frame to reduce CPU usage
            if frame_count % 3 == 0:
                # Convert the image from BGR color to RGB color
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if face_locations:
                    # Generate encodings for detected faces
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Process each detected face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(self.data["encodings"], face_encoding, tolerance=0.6)
                        name = "Unknown"
                        rectangle_color = (0, 0, 255)  # Red for unknown
                        
                        if True in matches:
                            matched_indexes = [i for (i, match) in enumerate(matches) if match]
                            name_counts = {}
                            for index in matched_indexes:
                                matched_name = self.data["names"][index]
                                name_counts[matched_name] = name_counts.get(matched_name, 0) + 1
                            
                            name = max(name_counts, key=name_counts.get)
                            rectangle_color = (0, 255, 0)  # Green for recognized
                            
                            # Mark attendance in database
                            self.db.mark_attendance(name)
                            
                        # Draw rectangle and name
                        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), rectangle_color, cv2.FILLED)
                        cv2.putText(frame, name, (left + 6, bottom - 6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Update the frame queue
                try:
                    self.frame_queue.put(frame, False)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, False)
                    except queue.Empty:
                        pass
                
            frame_count += 1

    def update_frame(self):
        """Update the display frame and attendance data"""
        if self.is_running:
            try:
                frame = self.frame_queue.get_nowait()
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            
            # Update attendance display and statistics
            self.update_attendance_display()
            self.update_statistics()
            
            # Schedule next update
            self.root.after(100, self.update_frame)

    def update_attendance_display(self):
        """Update the attendance display"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get today's attendance
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        records = self.db.get_daily_attendance(today)
        
        # Update tree
        for record in records:
            name, time, status = record
            status = status if status else "Absent"
            time = time if time else "---"
            self.tree.insert("", tk.END, values=(name, time, status))

    def update_statistics(self):
        """Update the statistics display"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        records = self.db.get_daily_attendance(today)
        
        total_students = len(self.db.get_all_students())
        present_students = sum(1 for record in records if record[2] == "Present")
        absent_students = total_students - present_students
        attendance_rate = (present_students / total_students * 100) if total_students > 0 else 0
        
        # Update statistics labels
        self.total_students_label.config(text=f"Total Students: {total_students}")
        self.present_students_label.config(text=f"Present: {present_students}")
        self.absent_students_label.config(text=f"Absent: {absent_students}")
        self.attendance_rate_label.config(text=f"Attendance Rate: {attendance_rate:.1f}%")
        
        # Update absent students list
        self.absent_list.delete(0, tk.END)
        for record in records:
            name, time, status = record
            if not status:  # If status is None or empty, student is absent
                self.absent_list.insert(tk.END, name)

    def export_to_csv(self):
        """Export attendance data to CSV"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance_{today}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Name", "Time", "Status"])  # Header
                
                for item in self.tree.get_children():
                    values = self.tree.item(item)['values']
                    writer.writerow(values)
                
            messagebox.showinfo("Success", f"Attendance exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export attendance: {str(e)}")

    def on_closing(self):
        """Handle application closing"""
        try:
            if messagebox.askokcancel("Quit", "Do you want to quit and reset the attendance database?"):
                if self.is_running:
                    self.stop_recognition()
                self.db.reset_attendance()
                self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error during cleanup: {str(e)}")
            self.root.destroy()

    def run(self):
        """Start the application"""
        self.update_attendance_display()
        self.update_statistics()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    try:
        app = IntegratedAttendanceSystem()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return 1
    return 0

if __name__ == "__main__":
    main()
