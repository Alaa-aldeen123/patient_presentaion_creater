from flask import Flask,render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys
import io
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from pptx import Presentation
from pptx.util import Inches
from ultralytics import YOLO
import torch


app = Flask(__name__)
app.secret_key = '12345'

# define all function that will be used

def create_new_presentation():
    pptx_resource = "main.pptx" # the presentation template
    with open(pptx_resource, "rb") as file:
        pptx_data = file.read()
    pptx_stream = io.BytesIO(pptx_data)
    new_presentation = Presentation(pptx_stream)

def remove_background(image):
    """
    Removes the background from an image saved in `folder_path`
    and overwrites the file with a white background.
    """
    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            input_image = Image.open(image)
            output_image = remove(input_image)

            # Make white background and paste the transparent result onto it
            white_background = Image.new("RGB", output_image.size, (255, 255, 255))
            if output_image.mode == "RGBA":
                white_background.paste(output_image, mask=output_image.split()[3])
            else:
                white_background.paste(output_image)

            white_background.save(image)
            return f"✅ Background removed: {image}"
        except Exception as e:
            return f"❌ Error processing {image}: {e}"
    else:
        return f"⚠️ Invalid file or file not found: {image}"

# great and show the user what is the app for
@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/doctor_info', methods=['GET', 'POST'])
def doctor_info():
    if request.method == 'POST':
        # Capture doctor info from form
        session['doc_name'] = request.form.get('doc_name')
        session['doc_email'] = request.form.get('doc_email')
        session['doc_phone'] = request.form.get('doc_phone')

        # Validate required fields
        if not all([session['doc_name'], session['doc_email'], session['doc_phone']]):
            return render_template('doctor_info.html', error="Please fill all required fields.")

        return redirect(url_for('patient_info'))

    return render_template('doctor_info.html', title='Doctor Information')

# get the patient information
@app.route('/patient_info', methods=['GET', 'POST'])
def patient_info():
    if request.method == 'POST':
        # Store the patient name
        session['patient_name'] = request.form.get('patient_name')

        # --- pre personal  ---
        pre_personal_front = request.files.get('pre_personal_front')
        if pre_personal_front and pre_personal_front.filename != '':
            filename = secure_filename(pre_personal_front.filename)
            session['pre_personal_front'] = filename
        else:
            session['pre_personal_front'] = None

        pre_personal_smile = request.files.get('pre_personal_smile')
        if pre_personal_smile and pre_personal_smile.filename != '':
            filename = secure_filename(pre_personal_smile.filename)
            session['pre_personal_smile'] = filename
        else:
            session['pre_personal_smile'] = None

        pre_personal_oblique = request.files.get('pre_personal_oblique')
        if pre_personal_oblique and pre_personal_oblique.filename != '':
            filename = secure_filename(pre_personal_oblique.filename)
            session['pre_personal_oblique'] = filename
        else:
            session['pre_personal_oblique'] = None

        pre_personal_profile = request.files.get('pre_personal_profile')
        if pre_personal_profile and pre_personal_profile.filename != '':
            filename = secure_filename(pre_personal_profile.filename)
            session['pre_personal_profile'] = filename
        else:
            session['pre_personal_profile'] = None

        # --- post personal  ---
        post_personal_front = request.files.get('post_personal_front')
        if post_personal_front and post_personal_front.filename != '':
            filename = secure_filename(post_personal_front.filename)
            session['post_personal_front'] = filename
        else:
            session['post_personal_front'] = None

        post_personal_smile = request.files.get('post_personal_smile')
        if post_personal_smile and post_personal_smile.filename != '':
            filename = secure_filename(post_personal_smile.filename)
            session['post_personal_smile'] = filename
        else:
            session['post_personal_smile'] = None

        post_personal_oblique = request.files.get('post_personal_oblique')
        if post_personal_oblique and post_personal_oblique.filename != '':
            filename = secure_filename(post_personal_oblique.filename)
            session['post_personal_oblique'] = filename
        else:
            session['post_personal_oblique'] = None

        post_personal_profile = request.files.get('post_personal_profile')
        if post_personal_profile and post_personal_profile.filename != '':
            filename = secure_filename(post_personal_profile.filename)
            session['post_personal_profile'] = filename
        else:
            session['post_personal_profile'] = None

        # --- pre arch  ---
        pre_arch_right = request.files.get('pre_arch_right')
        if pre_arch_right and pre_arch_right.filename != '':
            filename = secure_filename(pre_arch_right.filename)
            session['pre_arch_right'] = filename
        else:
            session['pre_arch_right'] = None

        pre_arch_front = request.files.get('pre_arch_front')
        if pre_arch_front and pre_arch_front.filename != '':
            filename = secure_filename(pre_arch_front.filename)
            session['pre_arch_front'] = filename
        else:
            session['pre_arch_front'] = None

        pre_arch_left = request.files.get('pre_arch_left')
        if pre_arch_left and pre_arch_left.filename != '':
            filename = secure_filename(pre_arch_left.filename)
            session['pre_arch_left'] = filename
        else:
            session['pre_arch_left'] = None

        pre_arch_upper = request.files.get('pre_arch_upper')
        if pre_arch_upper and pre_arch_upper.filename != '':
            filename = secure_filename(pre_arch_upper.filename)
            session['pre_arch_upper'] = filename
        else:
            session['pre_arch_upper'] = None

        pre_arch_lower = request.files.get('pre_arch_lower')
        if pre_arch_lower and pre_arch_lower.filename != '':
            filename = secure_filename(pre_arch_lower.filename)
            session['pre_arch_lower'] = filename
        else:
            session['pre_arch_lower'] = None

        # --- post arch  ---
        post_arch_right = request.files.get('post_arch_right')
        if post_arch_right and post_arch_right.filename != '':
            filename = secure_filename(post_arch_right.filename)
            session['post_arch_right'] = filename
        else:
            session['post_arch_right'] = None

        post_arch_front = request.files.get('post_arch_front')
        if post_arch_front and post_arch_front.filename != '':
            filename = secure_filename(post_arch_front.filename)
            session['post_arch_front'] = filename
        else:
            session['post_arch_front'] = None

        post_arch_left = request.files.get('post_arch_left')
        if post_arch_left and post_arch_left.filename != '':
            filename = secure_filename(post_arch_left.filename)
            session['post_arch_left'] = filename
        else:
            session['post_arch_left'] = None

        post_arch_upper = request.files.get('post_arch_upper')
        if post_arch_upper and post_arch_upper.filename != '':
            filename = secure_filename(post_arch_upper.filename)
            session['post_arch_upper'] = filename
        else:
            session['post_arch_upper'] = None

        post_arch_lower = request.files.get('post_arch_lower')
        if post_arch_lower and post_arch_lower.filename != '':
            filename = secure_filename(post_arch_lower.filename)
            session['post_arch_lower'] = filename
        else:
            session['post_arch_lower'] = None

        # --- pre cast  ---
        pre_cast_right = request.files.get('pre_cast_right')
        if pre_cast_right and pre_cast_right.filename != '':
            filename = secure_filename(pre_cast_right.filename)
            session['pre_cast_right'] = filename
        else:
            session['pre_cast_right'] = None

        pre_cast_front = request.files.get('pre_cast_front')
        if pre_cast_front and pre_cast_front.filename != '':
            filename = secure_filename(pre_cast_front.filename)
            session['pre_cast_front'] = filename
        else:
            session['pre_cast_front'] = None

        pre_cast_left = request.files.get('pre_cast_left')
        if pre_cast_left and pre_cast_left.filename != '':
            filename = secure_filename(pre_cast_left.filename)
            session['pre_cast_left'] = filename
        else:
            session['pre_cast_left'] = None

        pre_cast_upper = request.files.get('pre_cast_upper')
        if pre_cast_upper and pre_cast_upper.filename != '':
            filename = secure_filename(pre_cast_upper.filename)
            session['pre_cast_upper'] = filename
        else:
            session['pre_cast_upper'] = None

        pre_cast_lower = request.files.get('pre_cast_lower')
        if pre_cast_lower and pre_cast_lower.filename != '':
            filename = secure_filename(pre_cast_lower.filename)
            session['pre_cast_lower'] = filename
        else:
            session['pre_cast_lower'] = None

        # --- post cast  ---
        post_cast_right = request.files.get('post_cast_right')
        if post_cast_right and post_cast_right.filename != '':
            filename = secure_filename(post_cast_right.filename)
            session['post_cast_right'] = filename
        else:
            session['post_cast_right'] = None

        post_cast_front = request.files.get('post_cast_front')
        if post_cast_front and post_cast_front.filename != '':
            filename = secure_filename(post_cast_front.filename)
            session['post_cast_front'] = filename
        else:
            session['post_cast_front'] = None

        post_cast_left = request.files.get('post_cast_left')
        if post_cast_left and post_cast_left.filename != '':
            filename = secure_filename(post_cast_left.filename)
            session['post_cast_left'] = filename
        else:
            session['post_cast_left'] = None

        post_cast_upper = request.files.get('post_cast_upper')
        if post_cast_upper and post_cast_upper.filename != '':
            filename = secure_filename(post_cast_upper.filename)
            session['post_cast_upper'] = filename
        else:
            session['post_cast_upper'] = None

        post_cast_lower = request.files.get('post_cast_lower')
        if post_cast_lower and post_cast_lower.filename != '':
            filename = secure_filename(post_cast_lower.filename)
            session['post_cast_lower'] = filename
        else:
            session['post_cast_lower'] = None

        # --- pre x-rays  ---
        pre_panorama = request.files.get('pre_panorama')
        if pre_panorama and pre_panorama.filename != '':
            filename = secure_filename(pre_panorama.filename)
            session['pre_panorama'] = filename
        else:
            session['pre_panorama'] = None

        pre_cefalomitric = request.files.get('pre_cefalomitric')
        if pre_cefalomitric and pre_cefalomitric.filename != '':
            filename = secure_filename(pre_cefalomitric.filename)
            session['pre_cefalomitric'] = filename
        else:
            session['pre_cefalomitric'] = None

        pre_cefalomitric_tracing = request.files.get('pre_cefalomitric_tracing')
        if pre_cefalomitric_tracing and pre_cefalomitric_tracing.filename != '':
            filename = secure_filename(pre_cefalomitric_tracing.filename)
            session['pre_cefalomitric_tracing'] = filename
        else:
            session['pre_cefalomitric_tracing'] = None

        # --- post x-rays  ---
        post_panorama = request.files.get('post_panorama')
        if post_panorama and post_panorama.filename != '':
            filename = secure_filename(post_panorama.filename)
            session['post_panorama'] = filename
        else:
            session['post_panorama'] = None

        post_cefalomitric = request.files.get('post_cefalomitric')
        if post_cefalomitric and post_cefalomitric.filename != '':
            filename = secure_filename(post_cefalomitric.filename)
            session['post_cefalomitric'] = filename
        else:
            session['post_cefalomitric'] = None

        post_cefalomitric_tracing = request.files.get('post_cefalomitric_tracing')
        if post_cefalomitric_tracing and post_cefalomitric_tracing.filename != '':
            filename = secure_filename(post_cefalomitric_tracing.filename)
            session['post_cefalomitric_tracing'] = filename
        else:
            session['post_cefalomitric_tracing'] = None


        #return redirect(url_for('prosses'))
        return redirect(url_for('process-image'))
    return render_template('patient_info.html')

# show the progress
@app.route('/prosses', methods=['GET'])
def prosses():
    pass

# download the output
@app.route('/download', methods=['GET'])
def get_download():
    # create a new presentation
    create_new_presentation()
    remove_background(session['pre_personal_front'])


@app.route('/process-image')
def process_image():
    filename = session.get('pre_personal_front')
    if filename:
        remove_background(filename)
        return redirect(url_for('download_image', filename=filename))
    else:
        return "No image found in session."




if __name__=="__main__":
    app.run(debug=True)
