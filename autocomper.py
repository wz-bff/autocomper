import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import sv_ttk
from colorama import Fore, Style
from kthread import KThread
from PIL import Image, ImageTk
from proglog import ProgressBarLogger

from compile import compile_vid
from custom_tooltip import CustomHovertip
from file_utils import get_bundle_filepath
from sound_reader import get_timestamps

VIDEO_INPUT = [("Video Files",  "*.mp4 *.avi *.mkv *.m4v *.mov")]
VIDEO_OUTPUT = [("Video Files", "*.mp4"), ("All Files", "*.*")]
AUDIO_INPUT = [("Audio Files",  "*.mp3 *.wav *.flac")]
AUDIO_OUTPUT = [("Audio Files", "*.mp3"), ("All Files", "*.*")]


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Autocomper')

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set initial window size
        self.root.geometry('1000x800')

        # Enforce minimum window size
        self.root.resizable(False, True)

        # Create a grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # Left Column Widgets
        self.left_frame = ttk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=30, sticky="nsew")

        # Create a vertical separator
        separator = ttk.Separator(root, orient='vertical')
        separator.grid(row=0, column=1, sticky='ns')

        self.models_dir = "models/"

        self.precision = tk.IntVar(value=100)
        self.block_size = tk.IntVar(value=600)
        self.threshold = tk.DoubleVar(value=0.90)
        self.model = tk.StringVar()
        self.model.set("bdetectionmodel_05_01_23.onnx")
        self.merge_clips = tk.BooleanVar()
        self.combine_vids = tk.BooleanVar()
        self.normalize_audio = tk.BooleanVar()

        # Create a list to store uploaded video file paths
        self.uploaded_videos = []

        self.filelist_frame = ttk.Frame(self.left_frame)

        self.media_toggle_frame = ttk.Frame(self.filelist_frame)

        self.is_video = True

        def toggle_media():
            if self.is_video:
                self.toggle_button.config(text='Audio')
            else:
                self.toggle_button.config(text='Video')
            self.is_video = not self.is_video
            self.uploaded_videos.clear()
            self.update_listbox()
            self.clear_output()

        ttk.Label(self.media_toggle_frame, text="Input Media Type:",
                  font=(None, 12, "bold")).pack()

        self.toggle_button = ttk.Button(
            self.media_toggle_frame, text="Video", width=20, command=toggle_media)
        self.toggle_button.pack(pady=10)

        self.media_toggle_frame.pack(fill=tk.BOTH)

        self.video_listbox = ttk.Treeview(
            self.filelist_frame, selectmode=tk.EXTENDED, columns="#1", show='')
        self.video_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(self.filelist_frame, orient="vertical")
        scrollbar.config(command=self.video_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_listbox.config(yscrollcommand=scrollbar.set)

        self.filelist_frame.pack(fill=tk.BOTH)

        self.filelist_buttons_frame = ttk.Frame(self.left_frame)

        # Create buttons for adding and removing videos
        self.add_button = ttk.Button(
            self.filelist_buttons_frame, text="Add Media", command=self.add_video)
        self.up_arrow = ttk.Button(
            self.filelist_buttons_frame, text="↑", width=3, command=self.move_selected_up)
        self.down_arrow = ttk.Button(
            self.filelist_buttons_frame, text="↓", width=3, command=self.move_selected_down)

        self.remove_button = ttk.Button(
            self.filelist_buttons_frame, text="Remove Selected", command=self.remove_selected)
        self.clear_button = ttk.Button(
            self.filelist_buttons_frame, text="Clear All", command=self.clear_list)

        self.add_button.pack(pady=5, padx=1, side=tk.LEFT)
        self.up_arrow.pack(pady=5, padx=3, side=tk.LEFT)
        self.down_arrow.pack(pady=5, padx=3, side=tk.LEFT)

        self.clear_button.pack(pady=5, side=tk.RIGHT)
        self.remove_button.pack(pady=5, padx=5, side=tk.RIGHT)

        self.filelist_buttons_frame.pack(after=self.filelist_frame, fill=tk.X)

        ttk.Separator(self.left_frame, orient="horizontal").pack(
            fill=tk.X, pady=15)

        self.text_options_frame = ttk.Frame(self.left_frame)

        ttk.Label(self.text_options_frame, text="Model Options:",
                  font=(None, 11, "bold")).pack(pady=(10, 10))

        # Model Dropdown
        # First, get list of available models
        models = os.listdir(self.models_dir)

        # Filter out directories, keep only onnx files
        models = [item for item in models if os.path.isfile(
            os.path.join(self.models_dir, item))]

        models = [item for item in models if item.endswith('.onnx')]

        ttk.Label(self.text_options_frame, text="Model:", font=(
            None, 10, "bold")).pack(pady=(0, 1))

        self.model_dropdown = ttk.Combobox(
            self.text_options_frame, values=models, textvariable=self.model, state="readonly", width=30)

        self.model_dropdown.current(0)  # default dropdown option

        self.model_dropdown.pack()

        # Precision Entry
        ttk.Label(self.text_options_frame, text="Precision:",
                  font=(None, 10, "bold")).pack(pady=(10, 1))
        self.precision_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.precision)
        self.precision_entry.pack()

        # Block Size Entry
        ttk.Label(self.text_options_frame, text="Block Size (CAUTION):", font=(
            None, 10, "bold")).pack(pady=(10, 1))
        self.block_size_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.block_size)
        self.block_size_entry.pack()

        # Threshold Entry
        ttk.Label(self.text_options_frame, text="Threshold:",
                  font=(None, 10, "bold")).pack(pady=(10, 1))
        self.threshold_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.threshold)
        self.threshold_entry.pack()

        self.text_options_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        separator = ttk.Separator(self.left_frame, orient='vertical')
        separator.pack(side='left', fill='y', padx=(0, 15), pady=0)

        self.checkbox_frame = ttk.Frame(self.left_frame)
        self.checkbox_frame.pack(anchor=tk.W)

        ttk.Label(self.checkbox_frame, text="Video/Audio Options:",
                  font=(None, 11, "bold")).pack(pady=(10, 10), padx=0)

        # Merge Clips Checkbox
        self.merge_clips_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Merge Nearby Clips", variable=self.merge_clips)
        self.merge_clips_checkbox.pack(anchor=tk.W)

        # Merge Clips Checkbox
        self.combine_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Combine Input Media", variable=self.combine_vids, command=self.clear_output)
        self.combine_checkbox.pack(anchor=tk.W)

        # Normalize audio checkbox
        self.normalize_audio_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Normalize Audio", variable=self.normalize_audio)
        self.normalize_audio_checkbox.pack(anchor=tk.W)

        # Save timestamps to file checkbox
        self.save_txt = tk.BooleanVar()

        self.txt_output_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Save Timestamps to File", variable=self.save_txt)
        self.txt_output_checkbox.pack(anchor=tk.W)

        # Create a Checkbutton for custom resolution
        self.use_custom_resolution = tk.BooleanVar()

        self.custom_resolution_width_var = tk.IntVar()
        self.custom_resolution_width_var.set(1920)

        self.custom_resolution_height_var = tk.IntVar()
        self.custom_resolution_height_var.set(1080)

        self.checkbox_frame_three = ttk.Frame(self.left_frame)
        self.checkbox_frame_three.pack(anchor=tk.W)
        self.custom_resolution_checkbox = ttk.Checkbutton(
            self.checkbox_frame_three, text="Use Custom Output Resolution", variable=self.use_custom_resolution, command=self.toggle_text_boxes)
        self.custom_resolution_checkbox.pack(anchor=tk.W)

        self.container_frame = ttk.Frame(self.checkbox_frame_three)

        # Create text input boxes (initially hidden)
        self.res_width_label = ttk.Label(self.container_frame, text="Width:")
        self.res_width_entry = ttk.Entry(
            self.container_frame, textvariable=self.custom_resolution_width_var, width=5)

        self.res_height_label = ttk.Label(self.container_frame, text="Height:")
        self.res_height_entry = ttk.Entry(
            self.container_frame, textvariable=self.custom_resolution_height_var, width=5)

        self.res_width_label.pack(side=tk.LEFT)
        self.res_width_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.res_height_label.pack(side=tk.LEFT)
        self.res_height_entry.pack(side=tk.LEFT)

        # Right Column Widgets
        right_frame = ttk.Frame(root, width=300)
        right_frame.grid(row=0, column=2, padx=10, pady=30, sticky="nsew")
        right_frame.pack_propagate(False)

        self.output_video_path = tk.StringVar()
        self.output_video_path.set("No location selected!")

        # Output Location Selector
        ttk.Label(right_frame, text="Output Location:",
                  font=(None, 12, "bold")).pack()
        self.output_location_label = ttk.Label(
            right_frame, textvariable=self.output_video_path)
        self.output_location_label.pack(pady=10)
        self.output_location_button = ttk.Button(
            right_frame, text="Select Output File", width=34, command=self.select_output_location)
        self.output_location_button.pack(pady=(15, 2.5))

        self.process_cancel_frame = ttk.Frame(right_frame)
        self.process_cancel_frame.pack()

        # Process Video Button
        self.process_button = ttk.Button(
            self.process_cancel_frame, text='Process Videos', width=30, padding=4.5, command=self.process_videos_multi)
        self.process_button.grid(row=0, column=0, pady=(5, 20), padx=(0, 2.5))

        # Cancel Button
        stop_image_path = get_bundle_filepath(os.path.join("img", "stop.png"))
        stop_image = Image.open(stop_image_path).convert(mode='RGBA')
        stop_image = stop_image.resize((25, 25))
        stop_photo = ImageTk.PhotoImage(stop_image)

        self.cancel_button = ttk.Button(
            self.process_cancel_frame, image=stop_photo, width=5, padding=0, command=self.confirm_stop_process)
        self.cancel_button.image = stop_photo

        self.cancel_button.grid(row=0, column=1, pady=(5, 20), padx=(2.5, 0))

        # Progress bar for final render
        self.ui_bar = ttk.Progressbar(right_frame, orient='horizontal')
        self.ui_bar.pack(fill=tk.X, padx=10, pady=10)

        self.final_bar = FinalRenderBar(ui=self.ui_bar)

        self.stdout_frame = ttk.Frame(right_frame, width=200, height=100)

        # Text widget to display stdout
        self.stdout_text = tk.Text(
            self.stdout_frame, wrap="word", relief=tk.FLAT, fg="white")
        self.stdout_text.grid(row=0, column=0, sticky="nsew")

        text_scrollbar = ttk.Scrollbar(self.stdout_frame, orient="vertical")
        text_scrollbar.config(command=self.stdout_text.yview)
        text_scrollbar.grid(row=0, column=1, sticky="ns")

        self.stdout_text.config(yscrollcommand=text_scrollbar.set)

        # Configure grid weights to make the text widget expand
        self.stdout_frame.grid_rowconfigure(0, weight=1)
        self.stdout_frame.grid_columnconfigure(0, weight=1)

        self.stdout_frame.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout to the Text widget
        sys.stdout = StdoutRedirector(self.stdout_text)  # TODO: Change this

        self.active_thread = None

        # Tooltips galore
        prec_tooltip = CustomHovertip(
            self.precision_entry, 'Precision (in ms) of the timestamp selection process (higher is less precise)')
        block_tooltip = CustomHovertip(
            self.block_size_entry, 'Amount of seconds (of samples) to process at once.\nLarger sizes offer better performance, but will consume significantly more memory.\nWARNING: Setting this too high for very long videos will use up a LOT of memory; only turn this up if you know your computer can handle it.')
        thres_tooltip = CustomHovertip(
            self.threshold_entry, 'The confidence threshold for a sound to be reported from 0-1.')
        merge_tooltip = CustomHovertip(
            self.merge_clips_checkbox, 'If timestamps are close together, combine them into one longer clip')
        comb_tooltip = CustomHovertip(
            self.combine_checkbox, 'Combine everything into one output video.\nIf unchecked, you will instead select a directory, and output\nvideos will be saved as (original_title)_comped inside the directory.')
        res_tooltip = CustomHovertip(self.custom_resolution_checkbox,
                                     '(BUGGY) Sets the resolution of the output video(s).\nMost useful when combining videos\nof different resolutions. Only applicable if the input media is video.')
        norm_tooltip = CustomHovertip(
            self.normalize_audio_checkbox, 'Normalizes the audio of each clip to 0 dB. Use this if your clips have wildly different volumes.')
        output_tooltip = CustomHovertip(
            self.output_location_label, f"{self.output_video_path.get()}")
        cancel_tooltip = CustomHovertip(
            self.cancel_button, 'Cancel the current compilation process.')
        timestamps_tooltip = CustomHovertip(
            self.txt_output_checkbox, 'Save the timestamps to a txt file \'timestamps.txt\' in the output directory.')

        self.disable_while_processing = [
            self.add_button,
            self.remove_button,
            self.up_arrow,
            self.down_arrow,
            self.clear_button,
            self.process_button,
            # self.model_dropdown,
            self.precision_entry,
            self.block_size_entry,
            self.threshold_entry,
            self.merge_clips_checkbox,
            self.combine_checkbox,
            self.custom_resolution_checkbox,
            self.res_height_entry,
            self.res_width_entry,
            self.output_location_button,
            self.normalize_audio_checkbox,
            self.toggle_button,
            self.txt_output_checkbox
        ]

    def clear_output(self):
        self.output_video_path.set("No location selected!")
        self.output_tooltip = CustomHovertip(
            self.output_location_label, f"{self.output_video_path.get()}")

    def toggle_text_boxes(self):
        # Toggle the visibility of text boxes based on the checkbox state
        if self.use_custom_resolution.get():  # Checkbox is checked
            self.container_frame.pack(after=self.custom_resolution_checkbox)
        else:  # Checkbox is unchecked
            self.container_frame.pack_forget()

    def add_video(self):
        input_formats = VIDEO_INPUT if self.is_video else AUDIO_INPUT

        file_names = filedialog.askopenfilenames(
            filetypes=input_formats)
        if file_names:
            for file in file_names:
                self.uploaded_videos.append(file)
            self.update_listbox()

    def update_listbox(self):
        # Clear the listbox
        self.video_listbox.delete(*self.video_listbox.get_children())

        # Populate the listbox with uploaded videos
        for video_path in self.uploaded_videos:
            item_number = len(self.video_listbox.get_children())
            self.video_listbox.insert("", "end", item_number, values=(
                str(os.path.basename(video_path)).replace(" ", "\ ")))
        self.video_listbox.pack()

    def move_selected_up(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index and len(selected_index) != len(self.uploaded_videos):
            for i in sorted(selected_index):
                if i != 0:
                    self.uploaded_videos[i -
                                         1], self.uploaded_videos[i] = self.uploaded_videos[i], self.uploaded_videos[i - 1]
            self.update_listbox()

        self.video_listbox.selection_clear()
        for i in selected_index:
            if i != 0:
                self.video_listbox.selection_add(str(i - 1))
            else:
                self.video_listbox.selection_add(str(i))

    def move_selected_down(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index and len(selected_index) != len(self.uploaded_videos):
            for i in reversed(sorted(selected_index)):
                if i != len(self.uploaded_videos) - 1:
                    self.uploaded_videos[i], self.uploaded_videos[i +
                                                                  1] = self.uploaded_videos[i + 1], self.uploaded_videos[i]
            self.update_listbox()

        self.video_listbox.selection_clear()
        for i in selected_index:
            if i != len(self.uploaded_videos) - 1:
                self.video_listbox.selection_add(str(i + 1))
            else:
                self.video_listbox.selection_add(str(i))

    def remove_selected(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index:
            for i in reversed(sorted(selected_index)):
                del self.uploaded_videos[i]
            self.update_listbox()

    def clear_list(self):
        self.uploaded_videos = []
        self.update_listbox()

    def select_output_location(self):
        if self.combine_vids.get():
            output_formats = VIDEO_OUTPUT if self.is_video else AUDIO_OUTPUT
            file_name = filedialog.asksaveasfilename(
                defaultextension=".mp4", filetypes=output_formats)
            if file_name:
                self.output_video_path.set(file_name)
                self.output_tooltip = CustomHovertip(
                    self.output_location_label, f"{self.output_video_path.get()}")
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.output_video_path.set(folder_path)
                self.output_tooltip = CustomHovertip(
                    self.output_location_label, f"{self.output_video_path.get()}")

    def process_videos_multi(self):
        # Run video processing in new thread so the app doesn't hang
        self.active_thread = KThread(target=self.process_videos)
        self.active_thread.start()

    def is_thread_active(self):
        return type(self.active_thread) is KThread and self.active_thread.is_alive()

    def confirm_stop_process(self):
        # Check if there is a thread running
        if not self.is_thread_active():
            messagebox.showerror("Error", "No process is currently running!")
            return False
        else:
            confirm = messagebox.askyesno("Confirm Cancellation",
                                          f"The current job will be cancelled, losing all progress. Would you like to cancel?")
            if confirm:
                try:
                    self.active_thread.terminate()
                finally:
                    print(
                        f"\n{Fore.RED}FAILURE: Operation cancelled by user.")
                    for elt in self.disable_while_processing:
                        elt["state"] = tk.NORMAL
                    return True
            return False

    def on_closing(self):
        if self.is_thread_active():
            if self.confirm_stop_process():
                self.root.destroy()
        else:
            self.root.destroy()

    def process_videos(self):
        for elt in self.disable_while_processing:
            elt["state"] = tk.DISABLED

        try:
            precision = self.precision.get()
            block_size = self.block_size.get()
            threshold = self.threshold.get()
            selected_model = os.path.join(self.models_dir, self.model.get())
            merge_clips = self.merge_clips.get()
            combine = self.combine_vids.get()
            normalize = self.normalize_audio.get()
            save_timestamps = self.save_txt.get()

            # Get model location if in a compiled app
            selected_model = get_bundle_filepath(selected_model)

            self.stdout_text["state"] = tk.NORMAL
            self.stdout_text.delete("1.0", tk.END)
            self.stdout_text["state"] = tk.DISABLED
            self.root.update_idletasks()

            if not self.uploaded_videos:
                raise Exception("Please pick some videos to compile.")

            if not self.output_video_path.get() or self.output_video_path.get() == "No location selected!":
                raise Exception("Please specify an output location.")

            output_video_path = self.output_video_path.get()

            dict_list = []

            if combine and os.path.exists(output_video_path):
                if not messagebox.askyesno("Confirm Overwrite",
                                           f"Output file \'{output_video_path}\' already exists and will be overwritten. Would you like to continue?"):
                    raise (Exception("Operation cancelled."))

            if not combine:
                for video in self.uploaded_videos:
                    temp = str(video.split('/')[-1]).rsplit('.', 1)
                    temp = '.'.join(temp[:-1])
                    temp = str(output_video_path + '/' + temp + "_comped.mp4")
                    if os.path.exists(temp):
                        if not messagebox.askyesno("Confirm Overwrite",
                                                   f"Output file \'{video}\' already exists and will be overwritten. Would you like to continue?"):
                            raise (Exception("Operation cancelled."))

            res = ()
            if self.use_custom_resolution.get():
                res = (self.custom_resolution_width_var.get(),
                       self.custom_resolution_height_var.get())
            else:
                res = None

            # Set values for progress bar
            # If saving individually, or there is only one video
            if not combine or len(self.uploaded_videos) == 1:
                if self.is_video:
                    total_progress = 4 * len(self.uploaded_videos) * 100
                else:
                    total_progress = 2 * len(self.uploaded_videos) * 100

                self.final_bar.reset_total_progress(total_progress)
            else:
                if self.is_video:
                    total_progress = 4 * (len(self.uploaded_videos) + 1) * 100
                else:
                    total_progress = 2 * (len(self.uploaded_videos) + 1) * 100

                self.final_bar.reset_total_progress(total_progress)

            try:
                for i, input_video_path in enumerate(self.uploaded_videos):
                    print(
                        f"{Fore.GREEN}[{i + 1}/{len(self.uploaded_videos)}]{Style.RESET_ALL} Getting timestamps for {input_video_path.split('/')[-1]}")
                    timestamps = get_timestamps(
                        input_video_path, precision, block_size, threshold, 58, selected_model)
                    dict_list.append(timestamps)
                    num_found = len(timestamps['timestamps'])
                    if num_found > 1:
                        print(
                            f"{Fore.GREEN}Found {len(timestamps['timestamps'])} clips.")
                    elif num_found == 1:
                        print(
                            f"{Fore.GREEN}Found 1 clip.")
                    else:
                        self.final_bar.set_current_progress(
                            self.final_bar.current_progress + 100)
                        print(
                            f"{Fore.YELLOW}Could not find any clips.")

                # Save txt file with timestamp info
                if save_timestamps:
                    try:
                        if os.path.isdir(output_video_path):
                            txt_path = output_video_path
                        else:
                            txt_path = os.path.dirname(output_video_path)

                        def convert_seconds_to_timestamp(seconds: float) -> str:
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            remaining_seconds = int(
                                round((seconds % 3600) % 60))

                            if remaining_seconds == 60:
                                minutes += 1
                                remaining_seconds = 0

                            if minutes == 60:
                                hours += 1
                                minutes = 0

                            timestamp = f"{hours}:{minutes:02}:{remaining_seconds:02}"
                            return timestamp

                        timestamps_text = ""
                        for file in dict_list:
                            timestamps_text += f"{file['filename']}\n"

                            for ts in file['timestamps']:
                                timestamps_text += f"{convert_seconds_to_timestamp(ts['start'])}, confidence: {ts['pred']}\n"

                            timestamps_text += "\n"

                        with open(os.path.join(txt_path, "timestamps.txt"), 'w') as file:
                            file.write(timestamps_text)

                        print(f"{Fore.GREEN}Saved timestamps to timestamps.txt!")
                    except:
                        raise

                print(
                    f"Compiling and writing to {output_video_path.split('/')[-1]}...")
                compile_vid(dict_list, output_video_path, merge_clips,
                            combine, res, self.final_bar, normalize, self.is_video)
                print(
                    f"{Fore.GREEN}Wrote final video to {output_video_path.split('/')[-1]}.")
                messagebox.showinfo(
                    "Info", f"Video(s) exported to {output_video_path}. Enjoy!")
            except Exception as e:
                raise Exception(
                    "Encountered error during video processing: " + str(e))

            print(f"{Fore.GREEN}SUCCESS!")
            self.root.update_idletasks()

            for elt in self.disable_while_processing:
                elt["state"] = tk.NORMAL

        except Exception as e:
            messagebox.showerror("Error", e)
            print(f"\n{Fore.RED}FAILURE: " + str(e))
            for elt in self.disable_while_processing:
                elt["state"] = tk.NORMAL
            return


class StdoutRedirector:

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget["state"] = tk.DISABLED

    def write(self, text):
        self.text_widget["state"] = tk.NORMAL

        # Check/apply colorama colors
        # This is the worst code ever written but hey it works
        if Fore.RED in text:
            text = text.replace(Fore.RED, "")
            self.text_widget.tag_configure("red", foreground="red")
            self.text_widget.insert(tk.END, text, "red")
        elif Fore.YELLOW in text:
            text = text.replace(Fore.YELLOW, "")
            self.text_widget.tag_configure("yellow", foreground="yellow")
            self.text_widget.insert(tk.END, text, "yellow")
        elif Fore.GREEN in text:
            r, g, b = 144, 238, 144
            light_green = f"#{r:02x}{g:02x}{b:02x}"
            text = text.replace(Fore.GREEN, "")

            if Style.RESET_ALL in text:
                middle_index = text.find(Style.RESET_ALL)

                text = text.replace(Style.RESET_ALL, "")

                self.text_widget.insert(tk.END, text)

                start_index = self.text_widget.index("end-1c linestart")
                middle_index = self.text_widget.index(
                    f"{start_index}+{middle_index}c")

                self.text_widget.tag_configure(
                    light_green, foreground=light_green)
                self.text_widget.tag_configure(
                    "white", foreground="white")

                self.text_widget.tag_add(
                    light_green, start_index, middle_index)
                self.text_widget.tag_add("white", middle_index, tk.END)
            else:
                self.text_widget.tag_configure(
                    light_green, foreground=light_green)
                self.text_widget.insert(tk.END, text, light_green)
        else:
            self.text_widget.insert(tk.END, text)

        self.text_widget.see(tk.END)  # Scroll to the end of the text
        self.text_widget["state"] = tk.DISABLED

    def flush(self):
        return


class FinalRenderBar(ProgressBarLogger):
    def __init__(self, ui, init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0):
        self.ui = ui
        self.reset_total_progress(100)

        super().__init__(init_state, bars, ignored_bars,
                         logged_bars, min_time_interval, ignore_bars_under)

    def set_current_progress(self, current_progress):
        self.current_progress = current_progress

    def reset_total_progress(self, max_value):
        self.max_value = max_value
        self.current_progress = 0
        self.total_progress = 0

        self.ui['value'] = self.total_progress
        self.ui['maximum'] = self.max_value

    def callback(self, **changes):
        for (parameter, value) in changes.items():
            # print ('Parameter %s is now %s' % (parameter, value))
            return

    def bars_callback(self, bar, attr, value, old_value=None):
        self.current_progress = (value / self.bars[bar]['total']) * 100

        if self.current_progress >= 100:
            self.total_progress += self.current_progress
            self.current_progress = 0

        self.ui['value'] = self.total_progress + self.current_progress

        # percentage = (self.ui['value'] / self.max_value) * 100
        # print(f"{self.ui['value']}/{self.max_value} = {percentage}")
        # print(f"{value} vs. {self.bars[bar]['total']}")


def main():
    root = root = tk.Tk()
    sv_ttk.set_theme("dark")

    app = VideoProcessorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
