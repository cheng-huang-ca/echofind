# edge/

C++17 port of the DSP front end and the int8 model runner (workstream W5, session S5).
Planned layout: `include/echofind/`, `src/`, `bindings/` (pybind11), `bench/`, `CMakeLists.txt`.
Parity tests compare against golden vectors exported from `echofind.dsp`.
ARM timing runs on real ARM hardware, not in the x86_64 cloud VM.
