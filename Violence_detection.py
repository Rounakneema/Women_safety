import tensorflow as tf
import numpy as np
import cv2
import time
import argparse

# Load the trained model for violence detection
def ld_tf_md(md_pth="my_trained_Violence_model.joblib"):
    return tf.keras.models.load_model(md_pth)


def calc_dst(bx1, bx2):
    ctr1 = ((bx1[0] + bx1[2]) / 2, (bx1[1] + bx1[3]) / 2)
    ctr2 = ((bx2[0] + bx2[2]) / 2, (bx2[1] + bx2[3]) / 2)
    return np.linalg.norm(np.array(ctr1) - np.array(ctr2))

# Detect if any weapon.
def dtc_wpn(dtc, cls_nm, wpns=['knife', 'gun']):
    return any(cls_nm[int(dt[5])] in wpns for dt in dtc)

def anlz_hnds(frm, pst_est):
    img_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    hmn = pst_est.inference(img_rgb)
    if hmn:
        for h in hmn:
            lh = h.body_parts.get(4)  
            rh = h.body_parts.get(7)  
            if lh and rh and lh.y < 0.5 and rh.y < 0.5:
                return True
    return False

# Display labels on video frames with appropriate color codes
def dsp_lbl(frm, dt_lbl, x1, y1, lbl, colors):
    cv2.putText(frm, dt_lbl, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[lbl], 2)

def mntr_fghts(vdo_pth, dst_thr=100, mv_thr=50, emrg_dur=5, show_output=True):
    cp = cv2.VideoCapture(vdo_pth)
    md = ld_tf_md()  # Assuming model path is static
    pst_est = "my_trained_Violence_model.joblib"
    
    pr_dtc = []
    emrg_tr = False
    emrg_st_t = None
    emrg_cnt = 0
    dngr_cnt = 0
    frame_count = 0
    
    colors = {"Normal Situation": (255, 0, 0), "Dangerous Situation!": (0, 0, 255), "Emergency Detected!": (0, 0, 255)}
    
    while cp.isOpened():
        rt, frm = cp.read()
        if not rt:
            break

        frame_count += 1

        inpt = tf.convert_to_tensor(frm)
        inpt = inpt[tf.newaxis, ...]
        rslts = md(inpt)
        dtc = rslts['detection_boxes'].numpy()
        cls_nm = rslts['detection_classes'].numpy()

        psn_dtc = [d for i, d in enumerate(dtc) if int(cls_nm[i]) == 1]  # Filter for person class
        ag_hnd = anlz_hnds(frm, pst_est)  # Analyze hands positions

        fgt_dtc, dngr_dtc = False, False

        # Check for fights by calculating distances between people
        for i in range(len(psn_dtc)):
            for j in range(i + 1, len(psn_dtc)):
                dst = calc_dst(psn_dtc[i][:4], psn_dtc[j][:4])
                if dst < dst_thr:
                    fgt_dtc = True
                elif dst < dst_thr * 1.5 and len(psn_dtc) > 2:
                    dngr_dtc = True

        wpn_dtc = dtc_wpn(dtc, cls_nm)  # Check for weapons

        # Determine situation labels
        lbl = "Normal Situation"
        if fgt_dtc or (ag_hnd and wpn_dtc):
            emrg_cnt += 1
            lbl = "Emergency Detected!"
            emrg_tr = True
            emrg_st_t = time.time()
        elif dngr_dtc:
            dngr_cnt += 1
            lbl = "Dangerous Situation!"
        
        # Emergency detection based on past frames
        if emrg_cnt > 0 and (time.time() - emrg_st_t < emrg_dur):
            lbl = "Emergency Detected!"
        elif dngr_cnt >= 3:
            lbl = "Dangerous Situation!"

        # Display detection results and labels on video frames
        for i, dt in enumerate(dtc):
            x1, y1, x2, y2 = dt
            dt_lbl = f"{cls_nm[i]}: {rslts['detection_scores'][i]:.2f}"
            cv2.rectangle(frm, (int(x1), int(y1)), (int(x2), int(y2)), colors[lbl], 2)
            dsp_lbl(frm, dt_lbl, x1, y1, lbl, colors)

        # Display final situation label on the frame
        cv2.putText(frm, lbl, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[lbl], 3)

        if show_output:
            cv2.imshow('Violence Detection', frm)

        pr_dtc = psn_dtc.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cp.release()
    cv2.destroyAllWindows()

# Main execution through terminal or compiler
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection from Video")
    parser.add_argument('--video', help="Path to the input video", default=None)

    args = parser.parse_args()

    # If video path is provided via terminal, use that, else use default for running via compiler
    vdo_pth = args.video if args.video else "fv1.mp4"  # Default video path if no argument is provided
    
    # Call the main function
    mntr_fghts(vdo_pth, dst_thr=120, emrg_dur=8, show_output=True)
