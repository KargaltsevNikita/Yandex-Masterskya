from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from sefer_pipeline import (
    PipelineConfig,
    SeferPipeline,
    dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = APP_DIR / "best.pt"
DEFAULT_INPUT = r"C:\Users\nikka\Yandex Masterskya\Sefer\sefer_detector_workspace\dataset\test\images"
DEFAULT_OUTPUT = str(APP_DIR / "ocr_pipeline_output")

st.set_page_config(page_title="Сэфер — OCR переименование фото", layout="wide")
st.title("Сэфер — локальное переименование фото по номеру на табличке")

results_key = "sefer_results"
review_df_key = "sefer_review_df"
rename_log_key = "sefer_rename_log"
report_paths_key = "sefer_report_paths"
summary_key = "sefer_summary"
config_key = "sefer_last_config"
ready_for_apply_key = "sefer_ready_for_apply"

st.subheader("Параметры запуска")
center_left, center_right = st.columns(2)
with center_left:
    input_dir = st.text_input("Папка с фото", value=DEFAULT_INPUT, key="input_dir_main")
with center_right:
    output_dir = st.text_input("Папка с результатами", value=DEFAULT_OUTPUT, key="output_dir_main")

with st.sidebar:
    st.header("Настройки")
    st.write(f"**Весы детектора:** `{DEFAULT_WEIGHTS.name}`")

    st.subheader("Переименование")
    keep_original_name = st.checkbox("Сохранять исходное имя файла", value=False)
    user_prefix = st.text_input("Префикс перед номером", value="")
    apply_rename = st.checkbox("Подготовить копирование/перемещение файлов в renamed_files", value=False)
    rename_mode = st.radio("Режим файловых операций", options=["copy", "move"], horizontal=True, index=0)

    st.subheader("Порог уверенности")
    detect_conf = st.slider("Порог детекции", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
    ocr_high_conf = st.slider("Финальный порог OCR", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
    ocr_low_conf_trigger = st.slider("Порог включения fallback OCR", min_value=0.30, max_value=0.95, value=0.80, step=0.01)

    st.subheader("Опции OCR")
    save_debug_vis = st.checkbox("Сохранять debug-визуализации", value=False)
    save_crops = st.checkbox("Сохранять crop табличек", value=False)
    save_prepared = st.checkbox("Сохранять подготовленные варианты crop", value=False)

    st.subheader("Локальные модели PaddleOCR")
    paddle_det_model_dir = st.text_input("text_detection_model_dir", value="")
    paddle_rec_model_dir = st.text_input("text_recognition_model_dir", value="")
    paddle_cls_model_dir = st.text_input("textline_orientation_model_dir", value="")

run_clicked = st.button("Запустить обработку", type="primary", use_container_width=True)

if run_clicked:
    weights_path = DEFAULT_WEIGHTS
    if not weights_path.exists():
        st.error(f"Файл весов не найден рядом с приложением: {weights_path}")
        st.stop()

    cfg = PipelineConfig(
        detector_weights=str(weights_path),
        input_dir=input_dir,
        output_dir=output_dir,
        detect_conf=detect_conf,
        ocr_high_conf_threshold=ocr_high_conf,
        ocr_low_conf_trigger=ocr_low_conf_trigger,
        keep_original_name=keep_original_name,
        user_prefix=user_prefix,
        apply_rename=False,
        rename_mode=rename_mode,
        save_debug_vis=save_debug_vis,
        save_crops=save_crops,
        save_prepared_crops=save_prepared,
        paddle_det_model_dir=paddle_det_model_dir or None,
        paddle_rec_model_dir=paddle_rec_model_dir or None,
        paddle_cls_model_dir=paddle_cls_model_dir or None,
    )

    pipeline = SeferPipeline(cfg)
    progress = st.progress(0, text="Подготовка моделей…")
    status_box = st.empty()

    def on_progress(done: int, total: int, image_path: Path, status: str | None) -> None:
        progress.progress(done / total, text=f"Обработка: {done}/{total}")
        status_box.write(f"Текущий файл: `{image_path.name}` — статус: **{status or 'processing'}**")

    try:
        with st.spinner("Загрузка моделей и обработка изображений…"):
            df = pipeline.process_images_folder(progress_callback=on_progress)
            df = df.sort_values(by=["ocr_conf", "input_filename"], ascending=[True, True], na_position="first").reset_index(drop=True)
            report_paths = pipeline.save_reports(df, rename_log_df=None)
            summary = pipeline.build_summary(df)

        st.session_state[results_key] = df
        st.session_state[review_df_key] = df.copy()
        st.session_state[rename_log_key] = None
        st.session_state[report_paths_key] = {k: str(v) for k, v in report_paths.items()}
        st.session_state[summary_key] = summary
        st.session_state[config_key] = {
            **cfg.__dict__,
            "detector_weights": str(weights_path),
            "apply_rename": apply_rename,
        }
        st.session_state[ready_for_apply_key] = apply_rename
        progress.progress(1.0, text="Готово")
        status_box.success("Обработка завершена.")
    except Exception as e:
        progress.empty()
        status_box.empty()
        st.error(f"Ошибка во время обработки: {e}")

if results_key in st.session_state:
    df = st.session_state[results_key]
    rename_log_df = st.session_state.get(rename_log_key)
    summary = st.session_state[summary_key]
    report_paths = st.session_state[report_paths_key]

    st.subheader("Сводка")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего файлов", summary["n_files"])
    c2.metric("Распознано", summary["n_ok"])
    c3.metric("Распознано, но с сомнением", summary["n_low_conf"])
    c4.metric("Не распознано", summary["n_not_found"] + summary["n_ocr_invalid"])

    with st.expander("Использованные настройки", expanded=False):
        st.json(st.session_state.get(config_key, {}))

    st.subheader("Результаты")
    filters = st.multiselect(
        "Показать статусы",
        options=["ok", "low_conf", "not_found", "ocr_invalid"],
        default=["ok", "low_conf", "not_found", "ocr_invalid"],
    )
    shown_df = df[df["status"].isin(filters)].copy() if filters else df.copy()
    shown_df = shown_df.sort_values(by=["ocr_conf", "input_filename"], ascending=[True, True], na_position="first")
    st.dataframe(
        shown_df[
            [
                "input_filename", "ocr_clean_text", "ocr_conf", "det_conf",
                "status", "final_name", "ocr_variant", "det_message", "ocr_message"
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Выгрузка")
    export_df = st.session_state.get(review_df_key, df).copy()
    export_df = export_df.sort_values(by=["ocr_conf", "input_filename"], ascending=[True, True], na_position="first")
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button(
        "Скачать CSV",
        data=dataframe_to_csv_bytes(export_df),
        file_name="ocr_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    try:
        xlsx_bytes = dataframe_to_xlsx_bytes({
            "all_results": export_df,
            "ok": export_df[export_df["status"] == "ok"],
            "low_conf": export_df[export_df["status"] == "low_conf"],
            "manual_review": export_df[export_df["status"].isin(["not_found", "ocr_invalid"])],
            **({"rename_log": rename_log_df} if rename_log_df is not None else {}),
        })
        dl2.download_button(
            "Скачать XLSX",
            data=xlsx_bytes,
            file_name="ocr_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        dl2.info(f"XLSX недоступен: {e}")

    dl3.download_button(
        "Скачать JSON summary",
        data=json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ocr_report.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("Файлы на диске")
    st.code("\n".join(f"{k}: {v}" for k, v in report_paths.items()), language="text")

    review_df = st.session_state.get(review_df_key, df.copy())
    review_df = review_df.sort_values(by=["ocr_conf", "input_filename"], ascending=[True, True], na_position="first").reset_index(drop=True)
    review_mask = review_df["status"].isin(["low_conf", "not_found", "ocr_invalid"])
    review_candidates = review_df[review_mask].copy()

    st.subheader("Проверка сомнительных и нераспознанных файлов")
    if review_candidates.empty:
        st.success("Файлов для ручной проверки нет.")
    else:
        st.write("Здесь можно изменить итоговое имя файла для случаев с сомнительным или отсутствующим распознаванием.")

        edited_rows = []
        for idx, row in review_candidates.iterrows():
            with st.container(border=True):
                col_img, col_meta = st.columns([1, 2])
                with col_img:
                    image_path = Path(row["input_path"])
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True, caption=row["input_filename"])
                    else:
                        st.warning(f"Файл не найден: {row['input_filename']}")
                with col_meta:
                    status_map = {
                        "low_conf": "Распознано, но с сомнением",
                        "not_found": "Не распознано",
                        "ocr_invalid": "Не распознано",
                    }
                    st.markdown(f"**Статус:** {status_map.get(row['status'], row['status'])}")
                    st.markdown(f"**OCR:** `{row['ocr_clean_text'] or '—'}`")
                    st.markdown(f"**OCR confidence:** `{row['ocr_conf']:.3f}`")
                    st.markdown(f"**Текущее имя:** `{row['final_name']}`")

                    edited_name = st.text_input(
                        f"Новое имя для {row['input_filename']}",
                        value=row["final_name"],
                        key=f"rename_edit_{idx}",
                    )
                    edited_rows.append((idx, edited_name))

        if st.button("Применить изменения имён", use_container_width=True):
            updated_df = review_df.copy()
            for idx, edited_name in edited_rows:
                updated_df.at[idx, "final_name"] = edited_name.strip() if edited_name else updated_df.at[idx, "final_name"]
            st.session_state[review_df_key] = updated_df
            st.success("Изменения сохранены в таблице результатов.")

        current_review_df = st.session_state.get(review_df_key, review_df.copy())
        current_review_df = current_review_df.sort_values(by=["ocr_conf", "input_filename"], ascending=[True, True], na_position="first").reset_index(drop=True)
        current_manual = current_review_df[current_review_df["status"].isin(["low_conf", "not_found", "ocr_invalid"])].copy()
        st.dataframe(
            current_manual[["input_filename", "status", "ocr_conf", "final_name"]],
            use_container_width=True,
            hide_index=True,
        )

        if st.session_state.get(ready_for_apply_key, False):
            st.divider()
            st.write("После проверки можно скопировать или переместить файлы в папку `renamed_files`.")
            if st.button("Скопировать / переместить файлы по итоговым именам", type="primary", use_container_width=True):
                try:
                    cfg_dict = st.session_state.get(config_key, {}).copy()
                    cfg_dict["apply_rename"] = False
                    pipeline = SeferPipeline(PipelineConfig(**cfg_dict))
                    final_df = st.session_state.get(review_df_key, current_review_df).copy()
                    rename_log_df = pipeline.apply_rename_plan(final_df)
                    report_paths = pipeline.save_reports(final_df, rename_log_df=rename_log_df)
                    st.session_state[rename_log_key] = rename_log_df
                    st.session_state[report_paths_key] = {k: str(v) for k, v in report_paths.items()}
                    st.success("Файлы обработаны и сохранены в renamed_files.")
                except Exception as e:
                    st.error(f"Не удалось применить файловые операции: {e}")

    if rename_log_df is not None:
        st.subheader("Лог операций с файлами")
        st.dataframe(rename_log_df, use_container_width=True, hide_index=True)
else:
    st.warning("Здесь появятся результаты после запуска обработки.")
