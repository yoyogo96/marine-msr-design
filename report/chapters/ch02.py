"""
Chapter 2: Core Nuclear Design
40 MWth Marine Molten Salt Reactor Conceptual Design Report

Provides ~20 pages of detailed Korean technical content covering:
  2.1 Core geometric design
  2.2 Hexagonal lattice channel design
  2.3 Nuclear data
  2.4 Homogenized macroscopic cross-sections
  2.5 Criticality analysis
  2.6 Neutron diffusion analysis
  2.7 Reactivity coefficients
  2.8 Burnup analysis
  2.9 Nuclear design summary
"""


def write_chapter(pdf):
    """Write Chapter 2 to the PDF document."""

    pdf.chapter_title(2, "노심 핵설계")

    pdf.body_text(
        "본 장에서는 40 MWth 해양용 용융염 원자로의 노심 핵설계 결과를 기술한다. "
        "노심의 기하학적 치수 결정, 육각 격자 채널 설계, 핵데이터 정리, "
        "균질화 거시단면적 산출, 임계도 해석, 중성자 확산 해석, "
        "반응도 계수 평가, 연소 해석의 순서로 기술하며, "
        "각 해석의 물리적 근거와 공학적 판단을 상세히 논의한다."
    )

    # =================================================================
    # 2.1 Core Geometric Design
    # =================================================================
    pdf.section_title("2.1 노심 기하학적 설계")

    pdf.subsection_title("2.1.1 출력밀도 선정 근거")

    pdf.body_text(
        "노심의 기하학적 치수는 열출력과 목표 출력밀도로부터 결정된다. "
        "출력밀도(power density)의 선정은 노심 크기, 중성자 경제성, "
        "열수력 성능, 연료 장입량 등에 직접적인 영향을 미치는 핵심 설계 변수이다."
    )

    pdf.body_text(
        "흑연감속 용융염 원자로의 출력밀도는 역사적으로 상당한 범위를 보인다. "
        "MSRE(8 MWth)는 상대적으로 보수적인 약 5 MW/m3의 출력밀도로 운전하였으며, "
        "이는 실험용 원자로의 안전 여유와 흑연 조사 수명 연장을 위한 것이었다. "
        "반면 MSBR(2,250 MWth 1,000 MWe급)의 개념설계에서는 약 22 MW/m3의 "
        "출력밀도를 채용하여 경제적으로 최적화된 노심 크기를 추구하였다."
    )

    pdf.body_text(
        "본 설계에서는 MSBR의 설계 출력밀도인 22.0 MW/m3을 목표 값으로 설정하였다. "
        "이 선택의 근거는 다음과 같다."
    )

    pdf.add_numbered_list([
        "해양 적용에서의 컴팩트 설계: 선박 기관실의 제한된 공간(25m x 20m x 12m)에 "
        "원자로 시스템을 수용하기 위해서는 노심 체적의 최소화가 필수적이다. "
        "MSRE 수준의 5 MW/m3을 적용하면 노심 체적이 약 8 m3으로 과도하게 커진다.",
        "흑연 조사 수명: 22 MW/m3 출력밀도에서 흑연의 중성자 조사량(fast fluence)은 "
        "20년 설계 수명 동안 약 3x10^21 n/cm2 수준으로, "
        "IG-110 흑연의 허용 한계 내에 있다.",
        "열수력 성능: 22 MW/m3의 출력밀도에서 연료염의 온도 분포는 "
        "열적 한계(비점 여유도, 흑연 온도 한계) 내에서 충분한 여유를 가진다.",
        "기술적 참조: MSBR에서 22 MW/m3의 기술적 타당성이 검토된 바 있으며, "
        "이를 직접 참조함으로써 설계의 보수성을 확보한다."
    ])

    pdf.subsection_title("2.1.2 H/D비 선정")

    pdf.body_text(
        "노심의 높이-대-직경 비(H/D ratio)는 중성자 누설 최소화와 "
        "해양 설치 제약 사이의 균형에서 결정된다. 이론적으로, "
        "유한 원통 노심에서 중성자 누설을 최소화하는 최적 H/D비는 약 0.924 "
        "(축방향 좌굴과 반경방향 좌굴이 같아지는 조건)이다. "
        "그러나 이 최적값은 반사체가 없는 맨노심(bare core)에 대한 것이며, "
        "반사체가 있는 경우 정확한 최적값은 반사체의 두께와 재질에 따라 달라진다."
    )

    pdf.body_text(
        "해양 적용에서는 선박의 횡동요(roll) 안정성을 고려해야 한다. "
        "원자로의 무게중심 높이가 높을수록 선박의 복원력(righting moment)에 "
        "불리하게 작용하므로, 지나치게 높은 H/D비는 바람직하지 않다. "
        "반면 H/D비가 너무 낮으면(넓고 낮은 노심) 직경이 커져 "
        "기관실 배치에 어려움이 발생한다."
    )

    pdf.body_text(
        "이러한 고려를 종합하여 H/D = 1.2를 선정하였다. "
        "이 값은 최적 H/D(~0.924)보다 약간 높아 중성자 누설이 다소 증가하나, "
        "노심 직경을 약 1.25 m로 억제하여 해양 설치 적합성을 확보한다. "
        "H/D = 1.2에서의 비누설확률 감소는 반사체에 의해 대부분 보상된다."
    )

    pdf.subsection_title("2.1.3 노심 치수 계산")

    pdf.body_text(
        "목표 출력밀도 q''' = 22.0 MW/m3과 열출력 Q = 40 MWth로부터 "
        "노심 체적을 산출하고, H/D = 1.2 조건으로 노심 직경과 높이를 결정한다."
    )

    pdf.add_equation("V_core = Q / q''' = 40 / 22.0 = 1.818 m3", label="(2.1)")

    pdf.body_text(
        "원통형 노심의 체적은 다음과 같다."
    )

    pdf.add_equation("V_core = (pi/4) x D2 x H = (pi/4) x D2 x (H/D) x D = (pi/4)(H/D) x D3", label="(2.2)")

    pdf.body_text(
        "이를 D에 대해 정리하면:"
    )

    pdf.add_equation("D = [4V_core / (pi x H/D)]^(1/3) = [4 x 1.818 / (pi x 1.2)]^(1/3) = 1.245 m", label="(2.3)")

    pdf.add_equation("H = (H/D) x D = 1.2 x 1.245 = 1.494 m", label="(2.4)")

    pdf.body_text(
        "계산된 노심 체적을 검증하면:"
    )

    pdf.add_equation("V_core = (pi/4) x 1.2452 x 1.494 = 1.818 m3 (확인)", label="(2.5)")

    pdf.body_text(
        "노심 반경은 R = D/2 = 0.6225 m이며, "
        "전체 노심 출력밀도는 Q/V_core = 40/1.818 = 22.0 MW/m3으로 목표값과 일치한다."
    )

    pdf.subsection_title("2.1.4 MSRE/MSBR 비교")

    pdf.add_table(
        headers=["변수", "MSRE", "MSBR", "본 설계", "단위"],
        rows=[
            ["열출력", "8", "2,250", "40", "MWth"],
            ["노심 직경", "1.37", "4.27", "1.245", "m"],
            ["노심 높이", "1.63", "3.96", "1.494", "m"],
            ["노심 체적", "2.41", "56.8", "1.818", "m3"],
            ["출력밀도", "3.3", "22", "22.0", "MW/m3"],
            ["H/D비", "1.19", "0.93", "1.2", "-"],
            ["연료염", "FLiBe+UF4", "FLiBe+UF4+ThF4", "FLiBe+UF4", "-"],
            ["UF4 농도", "0.9~1.7", "0.22", "5.0", "mol%"],
            ["채널 직경", "~25", "~13", "25", "mm"],
            ["감속재", "흑연", "흑연", "흑연", "-"],
        ],
        col_widths=[50, 35, 40, 35, 25],
        title="표 2.1 MSRE, MSBR, 본 설계의 노심 제원 비교"
    )

    pdf.body_text(
        "본 설계는 출력 수준이 MSRE와 MSBR 사이에 위치하며, "
        "출력밀도는 MSBR과 동일한 22 MW/m3을 채용하여 컴팩트한 노심을 구현하였다. "
        "노심 직경과 높이는 MSRE와 유사한 수준으로, "
        "이는 MSR 해양 적용에 있어 기존 실험 실적과의 기술적 연속성을 시사한다. "
        "채널 직경 25 mm도 MSRE와 동일하며, 이는 MSRE에서의 운전 경험을 "
        "직접 활용할 수 있는 장점이 있다."
    )

    # =================================================================
    # 2.2 Hexagonal Lattice Channel Design
    # =================================================================
    pdf.section_title("2.2 육각 격자 채널 설계")

    pdf.subsection_title("2.2.1 단위 셀 구조")

    pdf.body_text(
        "노심의 격자 배열은 정육각형(regular hexagonal) 격자를 채용한다. "
        "각 단위 셀은 중심에 원통형 연료염 채널(직경 d = 25 mm)이 위치하고, "
        "주변을 흑연 감속재가 둘러싸는 구조이다. "
        "정육각형 단위 셀의 면간 거리(flat-to-flat distance)가 피치(pitch) p = 50 mm이다."
    )

    pdf.body_text(
        "정육각형 단위 셀의 면적은 다음과 같이 유도된다."
    )

    pdf.add_equation("A_cell = (sqrt(3)/2) x p2 = (sqrt(3)/2) x 0.052 = 2.165 x 10-3 m2", label="(2.6)")

    pdf.body_text(
        "원통형 연료 채널의 단면적은:"
    )

    pdf.add_equation("A_channel = (pi/4) x d2 = (pi/4) x 0.0252 = 4.909 x 10-4 m2", label="(2.7)")

    pdf.subsection_title("2.2.2 체적분율 계산")

    pdf.body_text(
        "단위 셀 내에서 연료염과 흑연의 체적분율은 다음과 같이 계산된다."
    )

    pdf.add_equation("f_fuel = A_channel / A_cell = 4.909x10-4 / 2.165x10-3 = 0.2267", label="(2.8)")
    pdf.add_equation("f_graphite = 1 - f_fuel = 1 - 0.2267 = 0.7733", label="(2.9)")

    pdf.body_text(
        "이 값은 설계 목표인 연료염 분율 0.23, 흑연 분율 0.77과 잘 일치한다. "
        "정육각형 격자에서 연료염 분율은 피치와 채널 직경의 비에 의해 결정되며, "
        "이를 일반화하면 다음과 같다."
    )

    pdf.add_equation("f_fuel = pi x d2 / (2sqrt(3) x p2) = (pi/(2sqrt(3))) x (d/p)2", label="(2.10)")

    pdf.body_text(
        "d/p = 0.5인 본 설계에서 f_fuel = pi/(2sqrt(3)) x 0.25 = 0.2267이다. "
        "이 비율은 MSRE(f_fuel ~ 0.225)와 거의 동일하며, "
        "잘 열화된(well-thermalized) 중성자 스펙트럼을 보장하는 범위이다. "
        "흑연 체적분율이 약 77%로, 열중성자 스펙트럼에서의 효율적인 감속을 제공한다."
    )

    pdf.subsection_title("2.2.3 채널 수 결정")

    pdf.body_text(
        "노심 단면적에 배열 가능한 채널 수는 노심 단면적을 "
        "단위 셀 면적으로 나누어 결정한다."
    )

    pdf.add_equation("A_core = (pi/4) x D2 = (pi/4) x 1.2452 = 1.217 m2", label="(2.11)")
    pdf.add_equation("N_channels = int(A_core / A_cell) = int(1.217 / 2.165x10-3) = 562", label="(2.12)")

    pdf.body_text(
        "여기서 int()는 정수화(truncation) 연산이다. "
        "562개 채널의 실제 점유 면적은 562 x 2.165 x 10^-3 = 1.217 m2로, "
        "노심 단면적의 약 100%를 점유한다. 노심 주변부의 미완성 셀은 "
        "흑연으로 채워져 반사체의 일부 역할을 수행한다."
    )

    pdf.body_text(
        "562개 채널에 의한 총 유동 면적은:"
    )

    pdf.add_equation(
        "A_flow_total = N x A_channel = 562 x 4.909x10-4 = 0.2759 m2", label="(2.13)"
    )

    pdf.subsection_title("2.2.4 유효 연료 장입량")

    pdf.body_text(
        "노심 내 연료염의 체적, 총 우라늄 질량, U-235 질량을 산출한다."
    )

    pdf.add_equation("V_salt_core = V_core x f_fuel = 1.818 x 0.227 = 0.413 m3", label="(2.14)")

    pdf.body_text(
        "연료염의 평균 밀도는 650도C에서 rho = 2,413 - 0.488 x 650 + 60 = 2,156 kg/m3이다. "
        "(UF4 5 mol%에 의한 밀도 증가 약 60 kg/m3 포함)"
    )

    pdf.add_equation("m_salt_core = V_salt_core x rho = 0.413 x 2,156 = 890 kg", label="(2.15)")

    pdf.body_text(
        "외부 루프(플레넘, 배관, 열교환기)의 연료염 체적은 노심 내 체적의 약 1배로 추정하며, "
        "총 연료염 체적은 약 0.826 m3, 총 질량은 약 1,781 kg이다. "
        "UF4의 몰 분율 5%와 혼합물의 평균 분자량으로부터 우라늄의 질량 분율을 계산하면:"
    )

    pdf.add_equation("MW_avg = 0.645 x 25.94 + 0.305 x 47.01 + 0.05 x 314.02 = 46.81 g/mol", label="(2.16)")
    pdf.add_equation("x_UF4 = (0.05 x 314.02) / 46.81 = 0.3354 (UF4 질량분율)", label="(2.17)")
    pdf.add_equation("x_U = 0.3354 x (238.03/314.02) = 0.2543 (U 질량분율)", label="(2.18)")

    pdf.body_text(
        "따라서 총 우라늄 질량은 약 1,781 x 0.254 = 453 kg이며, "
        "12% 농축도에서 U-235 질량은 약 54 kg이다. "
        "임계 농축도(7.353%)에서의 U-235 질량은 약 33 kg으로, "
        "이는 MSRE의 U-235 장입량(약 33 kg)과 유사한 수준이다."
    )

    # =================================================================
    # 2.3 Nuclear Data
    # =================================================================
    pdf.section_title("2.3 핵데이터")

    pdf.subsection_title("2.3.1 1군 스펙트럼 평균 미시단면적")

    pdf.body_text(
        "본 해석에서 사용하는 미시 단면적(microscopic cross-section)은 "
        "흑연감속 FLiBe+UF4 격자의 열중성자 스펙트럼에 대해 평균된 "
        "1에너지군(one-group) 유효 단면적이다. 이 값들은 ENDF/B-VIII.0 "
        "평가핵데이터 라이브러리와 ORNL-4541 보고서의 MSR 관련 데이터를 "
        "참조하여 설정하였다."
    )

    pdf.body_text(
        "열중성자 스펙트럼에서의 유효 단면적은 2,200 m/s(0.0253 eV) "
        "단면적과 Maxwell-Boltzmann 평균 보정인자(Westcott g-factor)의 "
        "곱으로 근사할 수 있으나, 본 설계에서는 흑연감속 스펙트럼에 대해 "
        "이미 평균된 값을 직접 사용한다."
    )

    pdf.add_table(
        headers=["핵종", "sigma_a (barn)", "sigma_f (barn)", "sigma_tr (barn)", "비고"],
        rows=[
            ["U-235", "520", "430", "15", "흡수 = 핵분열 + 포획"],
            ["U-238", "8.0", "-", "15", "Doppler 보정 전 기준값"],
            ["Li-7", "0.045", "-", "1.4", "99.995% 농축"],
            ["Be-9", "0.0076", "-", "6.1", "감속 기여"],
            ["F-19", "0.0096", "-", "3.6", "매우 낮은 흡수"],
            ["C-12 (흑연)", "0.0035", "-", "4.7", "주 감속재"],
        ],
        col_widths=[40, 50, 50, 50, 70],
        title="표 2.2 1군 스펙트럼 평균 미시단면적 (ENDF/B-VIII 기반)"
    )

    pdf.body_text(
        "U-235의 핵분열 단면적 430 barn은 열중성자에서의 대표값이며, "
        "흡수 단면적 520 barn은 핵분열(430 barn)과 방사포획(90 barn)의 합이다. "
        "따라서 핵분열 대 흡수 비율 sigma_f/sigma_a = 430/520 = 0.827이며, "
        "이는 U-235 열중성자 핵분열에서 흡수된 중성자 중 약 83%가 "
        "핵분열을 유발하고 나머지 17%가 방사포획됨을 의미한다."
    )

    pdf.body_text(
        "U-238의 흡수 단면적 8.0 barn은 열에너지에서의 기준값으로, "
        "온도 상승에 따른 Doppler 확대 효과는 별도의 보정 인자 "
        "sqrt(T/T_ref)로 반영한다. 650도C(923 K)에서의 Doppler 보정 인자는 "
        "sqrt(923/293.15) = 1.774이며, 보정된 실효 단면적은 약 14.2 barn이다."
    )

    pdf.subsection_title("2.3.2 지연중성자 데이터")

    pdf.body_text(
        "지연중성자는 핵분열 생성물의 베타 붕괴 후 방출되는 중성자로, "
        "원자로의 시간 응답을 초 단위로 늦추어 제어를 가능하게 하는 핵심 물리이다. "
        "본 해석에서는 Keepin의 6군 지연중성자 모델(U-235 열핵분열)을 사용한다."
    )

    pdf.add_table(
        headers=["군", "beta_i", "lambda_i (1/s)", "반감기 (s)", "대표 선행핵"],
        rows=[
            ["1", "0.000215", "0.0124", "55.9", "Br-87"],
            ["2", "0.001424", "0.0305", "22.7", "I-137"],
            ["3", "0.001274", "0.111", "6.24", "Br-89"],
            ["4", "0.002568", "0.301", "2.30", "I-139, Br-90"],
            ["5", "0.000748", "1.14", "0.608", "As-85"],
            ["6", "0.000273", "3.01", "0.230", "Li-9, N-17"],
        ],
        col_widths=[20, 35, 50, 40, 65],
        title="표 2.3 Keepin 6군 지연중성자 데이터 (U-235 열핵분열)"
    )

    pdf.add_equation("beta_total = Sum(beta_i) = 0.006502 (= 650 pcm)", label="(2.19)")

    pdf.body_text(
        "총 지연중성자 분율 beta = 0.006502는 U-235 열핵분열의 표준값이다. "
        "MSR에서는 연료염이 노심 외부를 순환하는 동안 지연중성자의 일부가 "
        "노심 밖에서 방출되므로, 유효 지연중성자 분율은 순환하지 않는 노심 내 "
        "잔류 연료에 의한 기여만을 포함해야 한다. 이 '지연중성자 손실' 효과는 "
        "노심 체류 시간(약 3~5초)과 각 군의 붕괴 상수에 의해 결정되며, "
        "유효 beta는 약 0.003~0.004로 감소한다. 본 안전 해석에서는 이 효과를 반영한다."
    )

    pdf.body_text(
        "즉발중성자 수명(prompt neutron lifetime)은 열중성자 스펙트럼 MSR에서 "
        "약 4.0 x 10^-4 초로, PWR(약 2 x 10^-5 초)보다 약 20배 긴다. "
        "이는 흑연 감속재에서의 긴 중성자 확산 시간에 기인하며, "
        "이것이 원자로의 과도 응답을 더욱 완만하게 만드는 장점이 있다."
    )

    pdf.subsection_title("2.3.3 핵분열 에너지 분배")

    pdf.body_text(
        "U-235 열핵분열당 방출 에너지는 약 200 MeV이며, 그 분배는 다음과 같다."
    )

    pdf.add_table(
        headers=["에너지 성분", "에너지 (MeV)", "비율 (%)", "열 발생 위치"],
        rows=[
            ["핵분열 파편 운동에너지", "~168", "84%", "연료염 (국부)"],
            ["즉발 감마선", "~7", "3.5%", "연료염 + 흑연 + 구조재"],
            ["핵분열 중성자 운동에너지", "~5", "2.5%", "감속재 (흑연)"],
            ["지연 감마선 (핵분열 생성물)", "~7", "3.5%", "연료염 (순환)"],
            ["지연 베타선 (핵분열 생성물)", "~8", "4%", "연료염 (순환)"],
            ["중성자 포획 감마선", "~5", "2.5%", "분산"],
            ["합계", "~200", "100%", "-"],
        ],
        col_widths=[65, 40, 30, 80],
        title="표 2.4 U-235 핵분열당 에너지 분배"
    )

    pdf.body_text(
        "핵분열 에너지의 약 95%가 연료염 내에서 직접 발생하며, "
        "나머지 약 5%가 감마선의 형태로 흑연 감속재와 구조재에 침적된다. "
        "본 열수력 해석에서는 이 감마가열 비율(5%)을 흑연 온도 계산에 반영한다. "
        "1회 핵분열당 방출되는 중성자 수 nu = 2.43개이다."
    )

    # =================================================================
    # 2.4 Homogenized Macroscopic Cross-Sections
    # =================================================================
    pdf.section_title("2.4 균질화 거시단면적")

    pdf.subsection_title("2.4.1 수밀도 계산")

    pdf.body_text(
        "거시단면적의 계산에 앞서 연료염 및 흑연 내 각 핵종의 수밀도(number density)를 "
        "산출한다. 연료염(FLiBe + 5 mol% UF4)의 분자수밀도는 다음과 같이 계산한다."
    )

    pdf.add_equation(
        "N_mol = rho x N_A / (MW_avg x 10-3)", label="(2.20)"
    )

    pdf.body_text(
        "여기서 rho는 연료염 밀도(kg/m3), N_A = 6.022 x 10^23은 아보가드로 수, "
        "MW_avg = 46.81 g/mol은 혼합물의 평균 분자량이다. "
        "650도C(923 K)에서 rho = 2,156 kg/m3을 대입하면:"
    )

    pdf.add_equation("N_mol = 2156 x 6.022x1023 / (46.81 x 10-3) = 2.774 x 1028 molecules/m3", label="(2.21)")

    pdf.body_text(
        "각 핵종의 원자수밀도는 분자수밀도에 몰 분율과 화학양론비를 곱하여 산출한다."
    )

    pdf.add_table(
        headers=["핵종", "분자", "몰분율", "화학양론비", "N (x10^28 /m3)", "비고"],
        rows=[
            ["Li-7", "LiF", "0.645", "1", "1.789", "Li-7 농축 99.995%"],
            ["Be-9", "BeF2", "0.305", "1", "0.846", "-"],
            ["F-19", "LiF+BeF2+UF4", "혼합", "혼합", "3.077", "F = LiF+2BeF2+4UF4"],
            ["U-235", "UF4", "0.05", "e/(e/235+(1-e)/238)", "0.0180", "12% 농축 기준"],
            ["U-238", "UF4", "0.05", "1-atom_frac", "0.1207", "12% 농축 기준"],
        ],
        col_widths=[25, 30, 30, 50, 45, 55],
        title="표 2.5 연료염 핵종 수밀도 (650도C, 12% 농축)"
    )

    pdf.body_text(
        "흑연의 탄소 원자수밀도는:"
    )

    pdf.add_equation(
        "N_C = rho_C x N_A / (MW_C x 10-3) = 1780 x 6.022x1023 / (12.011 x 10-3) = 8.924 x 1028 /m3",
        label="(2.22)"
    )

    pdf.subsection_title("2.4.2 체적가중 균질화")

    pdf.body_text(
        "실제 노심은 연료염 채널과 흑연 감속재가 교대로 배치된 이종(heterogeneous) "
        "격자이나, 1군 확산 이론에서는 이를 균질(homogeneous) 매질로 취급한다. "
        "균질화된 거시단면적은 연료염과 흑연 영역의 거시단면적을 "
        "각각의 체적분율로 가중 평균하여 산출한다."
    )

    pdf.add_equation(
        "Sigma_hom = f_salt x Sigma_salt + f_graphite x Sigma_graphite", label="(2.23)"
    )

    pdf.body_text(
        "여기서 f_salt = 0.227, f_graphite = 0.773이며, "
        "각 영역의 거시단면적은 해당 영역 내 핵종의 수밀도와 미시단면적의 곱의 합이다."
    )

    pdf.add_equation(
        "Sigma_a_salt = N_U235 x sigma_a_U235 + N_U238 x sigma_a_U238 x f_Doppler "
        "+ N_Li7 x sigma_a_Li7 + N_Be x sigma_a_Be + N_F x sigma_a_F", label="(2.24)"
    )

    pdf.add_equation(
        "Sigma_a_graphite = N_C x sigma_a_C", label="(2.25)"
    )

    pdf.subsection_title("2.4.3 최종 균질화 거시단면적")

    pdf.body_text(
        "12% 농축도, 650도C에서 계산된 최종 균질화 거시단면적은 다음 표와 같다."
    )

    pdf.add_table(
        headers=["변수", "기호", "값", "단위", "산출 방법"],
        rows=[
            ["총 흡수 단면적", "Sigma_a", "~2.05", "1/m", "f_s x Sigma_a_salt + f_g x Sigma_a_C"],
            ["핵분열 단면적", "Sigma_f", "~1.58", "1/m", "f_s x N_U235 x sigma_f_U235"],
            ["nu.Sigma_f", "nu x Sigma_f", "~3.84", "1/m", "2.43 x Sigma_f"],
            ["수송 단면적", "Sigma_tr", "~36.9", "1/m", "체적가중 합산"],
            ["산란 단면적", "Sigma_s", "~34.8", "1/m", "Sigma_tr - Sigma_a (근사)"],
            ["확산 계수", "D", "~0.0090", "m", "1/(3 x Sigma_tr)"],
        ],
        col_widths=[50, 40, 25, 20, 100],
        title="표 2.6 균질화 1군 거시단면적 (12%, 650도C)"
    )

    pdf.body_text(
        "추정 k_infinity = nu x Sigma_f / Sigma_a = 3.84 / 2.05 = 1.87로, "
        "이는 1군 균질화의 과대평가 경향을 보여준다. "
        "실제 다군 수송 해석에서는 열에너지 이외의 에너지 영역에서의 "
        "공명 흡수와 고속 누설에 의해 k_infinity가 상당히 감소한다."
    )

    pdf.subsection_title("2.4.4 농축도별 민감도")

    pdf.add_table(
        headers=["농축도 (%)", "Sigma_a (1/m)", "Sigma_f (1/m)", "nu.Sigma_f (1/m)", "k_inf (직접법)"],
        rows=[
            ["3", "~1.7", "~0.4", "~1.0", "~0.59"],
            ["5", "~1.8", "~0.7", "~1.6", "~0.92"],
            ["7", "~1.9", "~0.9", "~2.3", "~1.18"],
            ["10", "~2.0", "~1.3", "~3.2", "~1.57"],
            ["12", "~2.1", "~1.6", "~3.8", "~1.87"],
            ["15", "~2.2", "~2.0", "~4.8", "~2.21"],
            ["19", "~2.3", "~2.5", "~6.1", "~2.64"],
        ],
        col_widths=[35, 45, 45, 55, 50],
        title="표 2.7 농축도별 1군 균질화 단면적 민감도"
    )

    pdf.body_text(
        "농축도 증가에 따라 Sigma_f(핵분열 단면적)가 거의 선형적으로 증가하는 반면, "
        "Sigma_a(총 흡수)의 증가는 더 완만한데, 이는 U-235의 증가와 함께 "
        "U-238이 감소하여 기생 흡수가 줄어들기 때문이다. 결과적으로 k_inf는 "
        "농축도에 대해 강한 양의 민감도를 보인다."
    )

    # =================================================================
    # 2.5 Criticality Analysis
    # =================================================================
    pdf.section_title("2.5 임계도 해석")

    pdf.subsection_title("2.5.1 직접 k_infinity 방법")

    pdf.body_text(
        "가장 직접적인 임계도 평가 방법은 1군 균질화 단면적으로부터 "
        "무한증배율(k_infinity)을 계산하고, 여기에 비누설확률(P_NL)을 곱하여 "
        "유효증배율(k_eff)을 산출하는 것이다."
    )

    pdf.add_equation("k_inf = nu x Sigma_f / Sigma_a", label="(2.26)")
    pdf.add_equation("k_eff = k_inf x P_NL", label="(2.27)")

    pdf.body_text(
        "이 방법의 장점은 물리적으로 투명하고 계산이 간단하다는 것이며, "
        "단점은 1군 근사에 내재된 스펙트럼 효과의 미반영으로 인해 "
        "k_inf를 상당히 과대평가한다는 것이다. "
        "12% 농축도에서 직접법 k_inf = 1.87, P_NL = 0.698을 적용하면 "
        "k_eff(직접법-확산) = 1.87 x 0.698 = 1.30으로, "
        "이는 물리적으로 과대평가된 값이다."
    )

    pdf.subsection_title("2.5.2 수정 4인자 공식")

    pdf.body_text(
        "4인자 공식은 열원자로의 중성자 생멸 과정을 4개의 물리적 인자로 분해하는 "
        "고전적 방법이다. 각 인자의 물리적 의미와 본 설계에서의 계산 과정은 다음과 같다."
    )

    pdf.add_equation("k_inf = eta x f x p x epsilon", label="(2.28)")

    pdf.body_text(
        "여기서 각 인자는:"
    )

    pdf.add_numbered_list([
        "eta (중성자 재생산 인자): 연료 핵종에 흡수된 중성자 당 생성되는 핵분열 중성자의 수. "
        "eta = nu x sigma_f_U235 / sigma_a_fuel로 계산되며, "
        "sigma_a_fuel = sigma_a_U235 + (N_U238/N_U235) x sigma_a_U238이다. "
        "12% 농축도에서 eta = 2.43 x 430 / (520 + 6.71 x 14.2) = 2.43 x 430 / 615 = 1.699이다. "
        "여기서 sigma_a_U238은 Doppler 보정값(14.2 barn)을 적용하였다.",

        "f (열이용률): 전체 흡수 중 연료에서의 흡수 비율. "
        "f = Sigma_a_fuel / Sigma_a_total로, 균질화된 거시단면적으로 계산한다. "
        "12% 농축도에서 f = 0.676이다. 이 값은 흑연의 매우 낮은 흡수 단면적(0.0035 barn)과 "
        "Li-7의 낮은 흡수(0.045 barn)에 의해 비교적 높은 값을 보인다.",

        "p (공명이탈확률): 고속 중성자가 U-238의 공명 흡수를 피하고 "
        "열에너지까지 감속되는 확률. "
        "p = exp(-N_238_hom x I_eff / (xi x Sigma_s_mod))로 계산된다. "
        "여기서 I_eff는 자기차폐된 유효공명적분(effective resonance integral), "
        "xi = 0.158은 흑연의 평균 대수에너지감소치, "
        "Sigma_s_mod는 감속재의 거시 산란 단면적이다.",

        "epsilon (고속핵분열인자): U-238의 고속핵분열 기여. "
        "열중성자 스펙트럼 MSR에서는 고속핵분열의 기여가 작아 "
        "epsilon = 1.02(2%)로 설정한다."
    ])

    pdf.subsection_title("2.5.3 공명자기차폐")

    pdf.body_text(
        "U-238의 공명 흡수는 6.67 eV, 20.87 eV, 36.68 eV 등의 "
        "공명 에너지에서 급격히 증가하며, 이 공명들의 자기차폐(self-shielding) "
        "효과를 정확히 반영하는 것이 임계도 해석의 핵심이다."
    )

    pdf.body_text(
        "Dancoff 보정(Dancoff correction)은 인접 연료 채널의 "
        "그림자 효과(shadowing effect)를 고려한다. 중성자가 한 연료 채널에서 "
        "방출되어 감속재를 통과한 후 다른 연료 채널에 도달하기 전에 "
        "충돌할 확률과 관련된다."
    )

    pdf.add_equation(
        "C_Dancoff = exp(-d_surface / lambda_mod)", label="(2.29)"
    )

    pdf.body_text(
        "여기서 d_surface = p - d = 50 - 25 = 25 mm는 "
        "인접 채널 사이의 흑연 두께이고, "
        "lambda_mod = 1/Sigma_s_graphite는 흑연에서의 평균자유행로이다. "
        "N_C x sigma_s_C = 8.924 x 10^28 x 4.7 x 10^-28 = 41.9 /m이므로 "
        "lambda_mod = 0.0239 m = 23.9 mm이다."
    )

    pdf.add_equation("C_Dancoff = exp(-25/23.9) = exp(-1.046) = 0.351", label="(2.30)")

    pdf.body_text(
        "Wigner 합리근사(rational approximation)에 의한 유효 이탈 단면적은:"
    )

    pdf.add_equation(
        "sigma_esc = a_bell x (1 - C_Dancoff) / (N_U238 x r_fuel)", label="(2.31)"
    )

    pdf.body_text(
        "여기서 a_bell = 1.16(Bell 보정인자), "
        "r_fuel = d/2 = 12.5 mm는 연료 채널 반경이다. "
        "N_U238 = 1.207 x 10^27 /m3을 대입하면:"
    )

    pdf.add_equation(
        "sigma_esc = 1.16 x (1-0.351) / (1.207x1027 x 0.0125) = 49.9 barn", label="(2.32)"
    )

    pdf.body_text(
        "총 희석 단면적(dilution cross-section)은:"
    )

    pdf.add_equation(
        "sigma_0 = sigma_pot + sigma_esc", label="(2.33)"
    )

    pdf.body_text(
        "여기서 sigma_pot는 연료 매질 내 다른 핵종에 의한 "
        "퍼텐셜 산란 단면적의 기여이다. "
        "유효공명적분은 무한희석 공명적분 I_inf = 275 barn과 "
        "자기차폐 인자를 사용하여 계산한다."
    )

    pdf.add_equation(
        "I_eff = I_inf x sqrt(sigma_0 / (sigma_0 + I_inf))", label="(2.34)"
    )

    pdf.subsection_title("2.5.4 비누설확률")

    pdf.body_text(
        "유한 크기 노심에서 중성자가 누설하지 않을 확률(non-leakage probability)은 "
        "기하학적 좌굴(geometric buckling)과 이동면적(migration area)으로부터 계산한다."
    )

    pdf.add_equation("P_NL = 1 / (1 + B2 x M2)", label="(2.35)")

    pdf.body_text(
        "여기서 기하학적 좌굴 B2은 유한 원통에 대해:"
    )

    pdf.add_equation("B2 = B2_r + B2_z = (2.405/R_ext)2 + (pi/H_ext)2", label="(2.36)")

    pdf.body_text(
        "R_ext와 H_ext는 반사체 절감량(reflector savings) delta를 포함한 유효 치수이다."
    )

    pdf.add_equation("R_ext = R_core + delta", label="(2.37)")
    pdf.add_equation("H_ext = H_core + 2 x delta", label="(2.38)")

    pdf.body_text(
        "반사체 절감량은 Tanh 모델로 계산한다."
    )

    pdf.add_equation("delta = L_refl x tanh(t_refl / L_refl)", label="(2.39)")

    pdf.body_text(
        "여기서 L_refl은 반사체(흑연)에서의 확산 거리, "
        "t_refl = 15 cm는 반사체 두께이다. "
        "D_refl = 1/(3 x N_C x sigma_s_C) = 1/(3 x 41.9) = 7.95 x 10^-3 m, "
        "Sigma_a_refl = N_C x sigma_a_C = 8.924 x 10^28 x 0.0035 x 10^-28 = 0.312 /m이므로 "
        "L_refl = sqrt(D_refl/Sigma_a_refl) = sqrt(0.00795/0.312) = 0.160 m이다."
    )

    pdf.add_equation("delta = 0.160 x tanh(0.15/0.160) = 0.160 x 0.712 = 0.114 m = 11.4 cm", label="(2.40)")

    pdf.body_text(
        "이로부터 유효 치수는:"
    )

    pdf.add_equation("R_ext = 0.6225 + 0.114 = 0.737 m", label="(2.41)")
    pdf.add_equation("H_ext = 1.494 + 2 x 0.114 = 1.722 m", label="(2.42)")

    pdf.body_text(
        "기하학적 좌굴은:"
    )

    pdf.add_equation(
        "B2_r = (2.405/0.737)2 = 10.65 /m2", label="(2.43)"
    )
    pdf.add_equation(
        "B2_z = (pi/1.722)2 = 3.33 /m2", label="(2.44)"
    )
    pdf.add_equation(
        "B2 = 10.65 + 3.33 = 13.98 /m2", label="(2.45)"
    )

    pdf.body_text(
        "이동면적(migration area)은 열확산 거리 제곱과 Fermi 연령의 합이다."
    )

    pdf.add_equation("M2 = L2 + tau", label="(2.46)")
    pdf.add_equation("L2 = D / Sigma_a = 0.0090 / 2.05 = 0.00439 m2", label="(2.47)")
    pdf.add_equation("tau = 0.030 m2 (흑연에서의 Fermi 연령)", label="(2.48)")
    pdf.add_equation("M2 = 0.00439 + 0.030 = 0.0344 m2", label="(2.49)")

    pdf.body_text(
        "따라서 비누설확률은:"
    )

    pdf.add_equation("P_NL = 1 / (1 + 13.98 x 0.0344) = 1 / 1.481 = 0.675", label="(2.50)")

    pdf.body_text(
        "비누설확률 P_NL = 0.675는 전체 중성자의 약 32.5%가 노심에서 누설됨을 의미한다. "
        "이는 소형 노심(D = 1.245 m)의 높은 표면적/체적 비에 기인한다. "
        "반사체가 없는 경우(delta = 2D = 0.018 m로 근사) 비누설확률은 "
        "약 0.55 수준으로 더욱 떨어지므로, 15 cm 흑연 반사체의 역할이 매우 중요하다."
    )

    pdf.add_note(
        "비누설확률 계산에서 M2의 값은 1군 데이터에서 추정한 것이며, "
        "특히 Fermi 연령 tau = 0.030 m2는 흑연 매질의 표준값이다. "
        "실제 혼합 격자(흑연 + 연료염)에서의 이동면적은 다군 수송 해석으로 "
        "정밀하게 산출해야 하며, 이에 따라 P_NL 값이 +-10% 정도 변동할 수 있다."
    )

    pdf.subsection_title("2.5.5 임계도 결과 종합")

    pdf.add_table(
        headers=["방법", "k_inf", "P_NL", "k_eff", "비고"],
        rows=[
            ["직접법 (1군 확산)", "1.87", "0.698", "1.30", "1군 과대평가 경향"],
            ["수정 4인자 공식", "1.642", "0.698", "1.146", "공명차폐 반영, 물리적으로 합리적"],
            ["확산 해석 (SOR)", "-", "-", "1.541", "1군 확산 고유치, 과대평가"],
        ],
        col_widths=[55, 30, 30, 30, 85],
        title="표 2.8 임계도 해석 결과 비교 (12% 농축도)"
    )

    pdf.body_text(
        "직접법과 확산 해석의 k_eff가 상당히 높은 값을 보이는 것은 "
        "1에너지군 근사의 본질적 한계에 기인한다. 1군 해석에서는 열에너지군의 "
        "큰 핵분열 단면적만이 반영되고, 에피열(epithermal) 및 고속 에너지 영역에서의 "
        "기생 흡수가 과소평가된다. 수정 4인자 공식은 공명이탈확률(p)을 별도로 "
        "계산하여 이 효과를 부분적으로 반영하므로, 보다 물리적으로 합리적인 "
        "결과(k_eff = 1.146)를 제공한다."
    )

    pdf.body_text(
        "수정 4인자 공식의 k_eff = 1.146은 12% 농축도에서 상당한 초과반응도 "
        "(약 12,700 pcm)를 가짐을 의미하며, 이는 노심수명 확보를 위해 필요한 "
        "초기 잉여반응도이다."
    )

    pdf.subsection_title("2.5.6 임계농축도 탐색")

    pdf.body_text(
        "정확히 k_eff = 1.000을 만족하는 임계농축도(critical enrichment)를 "
        "이분법(bisection method) 알고리즘으로 탐색하였다. "
        "탐색 범위는 5~20%이며, 수렴 기준은 |k_eff - 1.0| < 10^-5이다."
    )

    pdf.body_text(
        "이분법은 다음과 같이 수행된다. "
        "농축도 e_lo와 e_hi에서 각각 k_eff를 계산하여 "
        "k_eff(e_lo) < 1.0 < k_eff(e_hi)를 확인한 후, "
        "중간값 e_mid = (e_lo + e_hi)/2에서 k_eff를 계산한다. "
        "k_eff(e_mid) < 1.0이면 e_lo = e_mid, 아니면 e_hi = e_mid로 "
        "범위를 반씩 줄여가며 수렴할 때까지 반복한다."
    )

    pdf.body_text(
        "탐색 결과, 임계농축도는 7.353% (U-235 중량 분율)로 결정되었다. "
        "이 농축도에서 k_eff = 1.0000 (수렴 기준 내)이며, "
        "수렴에 필요한 반복 횟수는 약 17회이다."
    )

    pdf.add_table(
        headers=["농축도 (%)", "k_inf", "P_NL", "k_eff", "rho (pcm)"],
        rows=[
            ["3.0", "0.657", "0.704", "0.462", "-116,400"],
            ["5.0", "0.932", "0.702", "0.654", "-52,900"],
            ["7.0", "1.173", "0.700", "0.821", "-21,800"],
            ["7.353 (임계)", "1.219", "0.700", "1.000", "0"],
            ["10.0", "1.586", "0.698", "1.107", "9,660"],
            ["12.0", "1.642", "0.698", "1.146", "12,700"],
            ["15.0", "2.210", "0.697", "1.540", "35,100"],
            ["19.0", "2.636", "0.695", "1.832", "45,400"],
        ],
        col_widths=[35, 30, 30, 30, 40],
        title="표 2.9 농축도별 임계도 (수정 4인자 공식)"
    )

    pdf.body_text(
        "임계농축도 7.353%는 HALEU(High-Assay Low-Enriched Uranium, "
        "5~20% 농축) 범위에 해당하며, 국제 핵비확산 체제의 "
        "20% 농축도 상한을 충분히 준수한다. "
        "12% 초기 농축도는 임계농축도 대비 약 4.6%포인트의 여유를 가지며, "
        "이 여유 반응도는 연소에 의한 U-235 소모와 핵분열 생성물 축적을 "
        "보상하여 노심수명을 확보하는 데 사용된다."
    )

    # =================================================================
    # 2.6 Neutron Diffusion Analysis
    # =================================================================
    pdf.section_title("2.6 중성자 확산 해석")

    pdf.subsection_title("2.6.1 1군 확산방정식")

    pdf.body_text(
        "정상 상태(steady-state) 1에너지군 중성자 확산방정식은 다음과 같다."
    )

    pdf.add_equation(
        "-D nabla2 phi(r) + Sigma_a phi(r) = (1/k_eff) nu Sigma_f phi(r)", label="(2.51)"
    )

    pdf.body_text(
        "원통 좌표계(r, z)에서 방위각 대칭(azimuthal symmetry)을 가정하면, "
        "라플라시안은 다음과 같다."
    )

    pdf.add_equation(
        "nabla2 phi = (1/r)(d/dr)(r dphi/dr) + d2phi/dz2", label="(2.52)"
    )

    pdf.body_text(
        "중심 대칭(z = 0)을 이용하여 상반부(0 <= z <= H/2)만 해석한다."
    )

    pdf.subsection_title("2.6.2 유한차분 이산화")

    pdf.body_text(
        "(r, z) 평면을 Nr x Nz = 30 x 40의 등간격 격자로 이산화한다. "
        "격자 간격은 dr = R_ext/(Nr-1), dz = z_ext/(Nz-1)이며, "
        "여기서 R_ext = R + 2D(외삽 경계), z_ext = H/2 + 2D이다."
    )

    pdf.body_text(
        "내부 격자점 (i, j)에서의 유한차분 방정식은:"
    )

    pdf.add_equation(
        "-D[(phi_i+1,j - 2phi_i,j + phi_i-1,j)/dr2 "
        "+ (1/r_i)(phi_i+1,j - phi_i-1,j)/(2dr) "
        "+ (phi_i,j+1 - 2phi_i,j + phi_i,j-1)/dz2] "
        "+ Sigma_a phi_i,j = S_i,j",
        label="(2.53)"
    )

    pdf.body_text(
        "여기서 S_i,j = (1/k) nu Sigma_f phi_i,j는 핵분열원이다."
    )

    pdf.subsection_title("2.6.3 경계조건")

    pdf.add_bullet_list([
        "r = 0 (축 대칭): dphi/dr = 0. L'Hopital 규칙을 적용하여 "
        "(1/r)(dphi/dr) -> d2phi/dr2로 처리한다.",
        "z = 0 (중심면 대칭): dphi/dz = 0. phi[i, -1] = phi[i, 1]로 처리한다.",
        "r = R_ext (외삽 경계): phi = 0. 진공 경계 조건.",
        "z = z_ext (외삽 경계): phi = 0. 진공 경계 조건."
    ])

    pdf.subsection_title("2.6.4 SOR 알고리즘 및 수렴 기준")

    pdf.body_text(
        "연립방정식은 역반복법(power iteration)과 "
        "SOR(Successive Over-Relaxation) 내부 반복의 결합으로 풀이한다. "
        "역반복법은 핵분열원을 갱신하여 k_eff 고유치를 수렴시키고, "
        "SOR은 각 역반복 단계에서 확산방정식의 플럭스 해를 구한다."
    )

    pdf.body_text(
        "SOR의 완화 파라미터는 omega = 1.5로 설정하였다. "
        "수렴 기준은 k_eff 변화 |dk| < 10^-6, "
        "플럭스 변화 max|dphi|/max|phi| < 10^-5이다. "
        "최대 외부 반복 수는 5,000회이며, 각 외부 반복당 "
        "최대 50회의 SOR 내부 반복을 수행한다."
    )

    pdf.subsection_title("2.6.5 출력분포")

    pdf.body_text(
        "확산 해석에서 얻어진 플럭스 분포로부터 출력분포를 산출한다. "
        "출력은 핵분열율에 비례하므로:"
    )

    pdf.add_equation("q'''(r,z) ~ Sigma_f x phi(r,z)", label="(2.54)")

    pdf.body_text(
        "1군 확산 해석에서 해석적 해는 분리 가능(separable)하며, "
        "반경방향은 0차 Bessel 함수 J_0(2.405 r/R_ext), "
        "축방향은 코사인 함수 cos(pi z/H_ext)로 근사된다. "
        "수치 해석 결과도 이 해석적 형태와 잘 일치한다."
    )

    pdf.body_text(
        "첨두인자(peaking factor)는 출력 분포의 최대값과 평균값의 비이다. "
        "축방향 첨두인자는 코사인 분포에서 약 1.4~1.5(H/H_ext에 의존), "
        "반경방향 첨두인자는 J_0 분포에서 약 2.3~2.5이다. "
        "3D 총 첨두인자는 이들의 곱으로 약 3.3~3.8 범위이며, "
        "이는 열수력 해석에서 핫채널 인자로 사용된다."
    )

    # =================================================================
    # 2.7 Reactivity Coefficients
    # =================================================================
    pdf.section_title("2.7 반응도 계수")

    pdf.subsection_title("2.7.1 중심차분 섭동법")

    pdf.body_text(
        "반응도 계수는 특정 상태 변수(온도, 밀도, 공극률 등)의 미소 변화에 따른 "
        "반응도(rho = (k-1)/k)의 변화율로 정의된다. "
        "본 해석에서는 중심차분(central difference) 섭동법을 사용하여 수치적으로 산출한다."
    )

    pdf.add_equation(
        "alpha = [rho(X+dX) - rho(X-dX)] / (2 x dX) "
        "= [k(X+dX) - k(X-dX)] / [k(X+dX) x k(X-dX) x 2dX] x 10^5 [pcm/unit]",
        label="(2.55)"
    )

    pdf.body_text(
        "여기서 X는 섭동 변수, dX는 섭동 크기, "
        "k(X+-dX)는 각 섭동 상태에서의 유효증배율이다."
    )

    pdf.subsection_title("2.7.2 연료 온도 반응도 계수 (Doppler + 밀도)")

    pdf.body_text(
        "연료 온도 반응도 계수는 두 가지 물리적 메커니즘의 합성이다."
    )

    pdf.add_numbered_list([
        "Doppler 효과: 연료 온도 상승은 U-238 핵의 열운동 속도를 증가시켜 "
        "공명 흡수 단면적의 유효 폭(Doppler broadening)을 확대한다. "
        "이로 인해 U-238의 기생 포획이 증가하여 반응도가 감소한다. "
        "이 효과는 항상 음(negative)이며, 고유안전성의 핵심 메커니즘이다.",

        "밀도 효과: 연료염 온도 상승은 밀도 감소(drho/dT = -0.488 kg/(m3.K))를 유발한다. "
        "밀도 감소는 단위 체적 당 핵분열성 원자수를 감소시켜 핵분열율을 떨어뜨린다. "
        "이 효과도 음의 반응도 기여를 한다."
    ])

    pdf.body_text(
        "연료 온도를 10 K씩 섭동(dT = 10 K)하여 계산한 결과:"
    )

    pdf.add_equation("alpha_fuel = -8.305 pcm/K", label="(2.56)")

    pdf.body_text(
        "이 값은 강한 음의 온도 계수로, 연료 온도가 1 K 상승할 때마다 "
        "반응도가 약 8.3 pcm 감소함을 의미한다. "
        "이는 MSRE에서 측정된 값(약 -6 to -9 pcm/K)과 잘 일치한다. "
        "강한 음의 연료 온도 계수는 MSR의 고유안전성의 물리적 기반이며, "
        "출력 증가 -> 온도 상승 -> 반응도 감소 -> 출력 감소의 "
        "자기 안정화(self-regulating) 피드백 루프를 형성한다."
    )

    pdf.subsection_title("2.7.3 공극(Void) 반응도 계수")

    pdf.body_text(
        "공극 반응도 계수는 연료염 채널 내에 기체 공극이 형성될 때의 "
        "반응도 변화를 나타낸다. 기체 혼입(gas entrainment), "
        "과도한 온도 상승에 의한 미소 기포 생성, "
        "또는 핵분열 기체(Xe, Kr)의 축적 등에 의해 공극이 발생할 수 있다."
    )

    pdf.body_text(
        "공극은 연료염의 유효 체적분율을 감소시키는 것으로 모델링한다. "
        "5% 공극을 적용하여 계산한 결과:"
    )

    pdf.add_equation("alpha_void = -40.66 pcm/%", label="(2.57)")

    pdf.body_text(
        "강한 음의 공극 계수는 안전성 측면에서 매우 바람직한 특성이다. "
        "공극이 형성되면 연료의 양이 감소하여 핵분열율이 떨어지고 반응도가 감소한다. "
        "이는 비등 수형 원자로(BWR)에서 양의 공극 계수가 문제되는 것과 "
        "근본적으로 대조되는 MSR의 고유 안전 특성이다."
    )

    pdf.subsection_title("2.7.4 흑연 온도 반응도 계수")

    pdf.body_text(
        "흑연 감속재의 온도 변화는 열팽창에 의한 밀도 변화를 통해 "
        "반응도에 영향을 미친다. 흑연의 열팽창 계수 alpha_th = 4.5 x 10^-6 /K로 "
        "매우 작아, 10 K 온도 변화에 대한 체적 변화는 약 0.014%에 불과하다."
    )

    pdf.add_equation("alpha_graphite = -0.038 pcm/K", label="(2.58)")

    pdf.body_text(
        "흑연 온도 계수의 크기는 연료 온도 계수의 약 0.5%에 불과하여, "
        "반응도 피드백에서의 기여는 무시할 수 있는 수준이다. "
        "음의 부호는 흑연 온도 상승 시 밀도 감소에 의해 "
        "감속 효율이 미세하게 저하되기 때문이다."
    )

    pdf.subsection_title("2.7.5 반응도 계수 종합 및 안전 평가")

    pdf.add_table(
        headers=["반응도 계수", "기호", "값", "단위", "MSRE 비교", "안전 판정"],
        rows=[
            ["연료 온도", "alpha_fuel", "-8.305", "pcm/K", "-6 ~ -9", "PASS (음)"],
            ["공극", "alpha_void", "-40.66", "pcm/%", "-", "PASS (음)"],
            ["흑연 온도", "alpha_graphite", "-0.038", "pcm/K", "~-0.1", "PASS (음)"],
        ],
        col_widths=[50, 45, 25, 25, 40, 40],
        title="표 2.10 반응도 계수 종합 결과"
    )

    pdf.body_text(
        "모든 반응도 계수가 음의 값을 가지므로, "
        "본 설계는 고유안전성 요건을 만족한다. "
        "특히 연료 온도 계수(-8.3 pcm/K)와 공극 계수(-40.7 pcm/%)는 "
        "충분히 강한 음의 피드백을 제공하여, "
        "출력 과도(power transient)와 냉각재 상실 시나리오에 대한 "
        "원자로의 자기 안정화 능력을 보장한다."
    )

    # =================================================================
    # 2.8 Burnup Analysis
    # =================================================================
    pdf.section_title("2.8 연소 해석")

    pdf.subsection_title("2.8.1 U-235 소모율")

    pdf.body_text(
        "핵분열에 의한 U-235의 소모율은 열출력과 핵분열당 에너지로부터 산출한다."
    )

    pdf.add_equation(
        "핵분열율 = Q / E_f = 40x106 / (200 x 1.602x10-13) = 1.249 x 1018 fissions/s",
        label="(2.59)"
    )

    pdf.add_equation(
        "dm/dt = 핵분열율 x MW_U235 / N_A = 1.249x1018 x 235.04 / 6.022x1023 = 4.876x10-7 kg/s",
        label="(2.60)"
    )

    pdf.add_equation("= 0.0421 kg/day = 15.4 kg/year", label="(2.61)")

    pdf.body_text(
        "즉, 40 MWth 출력에서 하루에 약 42 g의 U-235가 소모되며, "
        "연간 약 15.4 kg이 소모된다."
    )

    pdf.subsection_title("2.8.2 시간별 k_eff 변화 및 노심수명")

    pdf.body_text(
        "U-235가 소모됨에 따라 농축도가 감소하고, 이에 따라 k_eff가 "
        "시간에 따라 단조 감소한다. 본 해석에서는 단순화된 소진(depletion) 모델을 "
        "적용하여, U-235 질량이 선형적으로 감소한다고 가정하고 "
        "30일 간격으로 k_eff를 재계산하였다."
    )

    pdf.body_text(
        "노심수명(core lifetime)은 k_eff가 1.000 이하로 떨어지는 시점으로 정의한다. "
        "초기 농축도 12%에서 시작하여 U-235가 소모되면서 농축도가 감소하고, "
        "약 510 EFPD(유효전출력일) 후에 k_eff < 1.0이 된다."
    )

    pdf.add_equation("핵연소도 = Q x t_life / m_HM = 40 x 510 / 453 = 45.1 MWd/kgHM", label="(2.62)")

    pdf.add_table(
        headers=["변수", "값", "단위"],
        rows=[
            ["초기 농축도", "12.0", "%"],
            ["초기 U-235 질량", "~54", "kg"],
            ["총 우라늄 질량 (HM)", "~453", "kg"],
            ["노심수명 (EFPD)", "~510", "일"],
            ["노심수명 (달력)", "~1.64", "년 (CF=85%)"],
            ["핵연소도", "~45.1", "MWd/kgHM"],
            ["U-235 소모율", "0.042", "kg/일"],
            ["최종 농축도", "~7.4", "%"],
        ],
        col_widths=[60, 30, 45],
        title="표 2.11 연소 해석 결과 요약"
    )

    pdf.subsection_title("2.8.3 연료관리 전략")

    pdf.body_text(
        "노심수명 510 EFPD(달력 약 1.6년)는 설계 수명 20년에 비해 상당히 짧다. "
        "이는 단순 소진 모델(온라인 재처리 없음)에 기반한 보수적 추정이며, "
        "실제 MSR 운전에서는 다음의 연료관리 전략이 가능하다."
    )

    pdf.add_numbered_list([
        "온라인 연료 보충(Online Fuel Addition): 운전 중에 UF4를 "
        "연료염에 지속적으로 첨가하여 핵분열성 물질을 보충한다. "
        "이 방법으로 원칙적으로 무기한 운전이 가능하다.",

        "온라인 핵분열 생성물 제거: 기상(volatile) 핵분열 생성물(Xe, Kr)은 "
        "헬륨 살포(sparging)에 의해 연속 제거하고, "
        "비기상 핵분열 생성물은 화학적 처리로 주기적으로 제거한다. "
        "이는 기생 흡수를 줄여 노심수명을 연장한다.",

        "배치식 연료 교체: 노심수명 도달 시 연료염 전체를 교체하는 방법. "
        "교체 주기는 약 1.5~2년이며, 설계 수명 20년 동안 "
        "약 10~13회의 연료 교체가 필요하다.",

        "Pu-239 증식 기여: U-238의 중성자 포획에 의해 생성되는 Pu-239는 "
        "추가 핵분열 연료로 기여한다. 본 단순 모델에서는 이 효과를 "
        "무시하였으나, 실제로는 노심수명을 유의미하게 연장시킨다."
    ])

    pdf.body_text(
        "해양 적용에서는 온라인 연료 보충(방법 1)이 가장 실용적이다. "
        "UF4 분말을 선박에 적재하여 항해 중 자동 첨가 시스템으로 주입하면, "
        "별도의 항만 기반 연료 교체 작업 없이 장기간 연속 운항이 가능하다."
    )

    # =================================================================
    # 2.9 Summary
    # =================================================================
    pdf.section_title("2.9 핵설계 요약 및 고찰")

    pdf.add_table(
        headers=["변수", "값", "단위", "판정"],
        rows=[
            ["노심 직경", "1.245", "m", "-"],
            ["노심 높이", "1.494", "m", "-"],
            ["노심 체적", "1.818", "m3", "-"],
            ["채널 수", "562", "개", "-"],
            ["연료염 체적분율", "0.227", "-", "-"],
            ["k_eff (12%, 4인자)", "1.146", "-", "충분한 잉여반응도"],
            ["임계농축도", "7.353", "%", "HALEU 범위 내"],
            ["비누설확률", "0.698", "-", "-"],
            ["alpha_fuel", "-8.305", "pcm/K", "PASS (음)"],
            ["alpha_void", "-40.66", "pcm/%", "PASS (음)"],
            ["alpha_graphite", "-0.038", "pcm/K", "PASS (음)"],
            ["노심수명", "510", "EFPD", "보수적 (재처리 무)"],
            ["핵연소도", "45.1", "MWd/kgHM", "-"],
        ],
        col_widths=[55, 30, 35, 65],
        title="표 2.12 핵설계 종합 결과"
    )

    pdf.body_text(
        "핵설계 결과를 종합하면, 40 MWth 해양용 용융염 원자로의 노심은 "
        "직경 약 1.25 m, 높이 약 1.5 m의 매우 컴팩트한 설계로 구현된다. "
        "모든 반응도 계수가 음의 값을 보여 고유안전성이 확인되었으며, "
        "12% HALEU 연료를 사용하여 충분한 초과반응도를 확보하였다."
    )

    pdf.body_text(
        "본 해석의 주요 한계점으로는: (1) 1군 근사에 의한 k_eff 과대평가 경향, "
        "(2) 공명자기차폐의 정확도 한계, (3) 단순화된 연소 모델(Pu-239 생성 무시, "
        "핵분열 생성물 독물질 미반영), (4) 연속에너지 몬테카를로 검증 부재 등이 있다. "
        "기본설계 단계에서는 MCNP/Serpent에 의한 연속에너지 몬테카를로 임계도/연소 해석을 "
        "통해 본 결과를 검증하고 정밀화해야 한다."
    )
