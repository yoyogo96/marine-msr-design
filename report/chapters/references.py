"""
References and Appendix: 참고문헌 및 부록
==========================================

40 MWth 해양용 용융염 원자로 개념설계 보고서의 참고문헌 목록과
부록 A(설계 매개변수 종합표)를 포함한다.

Note: This module uses chapter_title without a number for the
references section. It creates two sections: references and appendix.

write_chapter(pdf) 함수는 MSRReport 인스턴스를 받아
참고문헌 및 부록 내용을 PDF에 기록한다.
"""


def write_chapter(pdf):
    """Write References and Appendix A."""

    # ===================================================================
    # 참고문헌
    # ===================================================================
    # Use a new page with custom heading (since there's no chapter number)
    pdf.add_page()
    pdf._current_chapter = "참고문헌"
    pdf._toc_entries.append((0, None, "참고문헌", pdf.page_no()))

    # Title
    pdf.set_font("AppleSD", "B", 18)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(w=0, h=10, text="참고문헌")
    pdf.ln(4)

    # Decorative line
    pdf.set_draw_color(26, 54, 93)
    pdf.set_line_width(1.0)
    y = pdf.get_y()
    pdf.line(25, y, 65, y)
    pdf.set_line_width(0.3)
    pdf.line(67, y, 190, y)
    pdf.ln(10)

    pdf.set_text_color(45, 55, 72)

    # References - using numbered list style but with formatted entries
    references = [
        # 1-5: ORNL MSR Reports
        "[1]  Robertson, R.C. (ed.), \"Conceptual Design Study of a Single-Fluid "
        "Molten-Salt Breeder Reactor,\" ORNL-4541, Oak Ridge National Laboratory, "
        "1971.",

        "[2]  Thoma, R.E. (ed.), \"Phase Diagrams of Nuclear Reactor Materials,\" "
        "ORNL-2548, Oak Ridge National Laboratory, 1959. (FLiBe phase diagram "
        "referenced via ORNL-TM-0728)",

        "[3]  Haubenreich, P.N. and Engel, J.R., \"Experience with the Molten-Salt "
        "Reactor Experiment,\" Nuclear Applications and Technology, Vol. 8, No. 2, "
        "pp. 118-136, 1970.",

        "[4]  Williams, D.F., Toth, L.M., and Clarno, K.T., \"Assessment of "
        "Candidate Molten Salt Coolants for the Advanced High Temperature Reactor "
        "(AHTR),\" ORNL/TM-2006/12, Oak Ridge National Laboratory, 2006.",

        "[5]  MacPherson, H.G., \"The Molten Salt Reactor Adventure,\" Nuclear "
        "Science and Engineering, Vol. 90, No. 4, pp. 374-380, 1985.",

        # 6-10: Regulatory and Standards
        "[6]  International Commission on Radiological Protection, \"The 2007 "
        "Recommendations of the International Commission on Radiological "
        "Protection,\" ICRP Publication 103, Annals of the ICRP, Vol. 37, "
        "Nos. 2-4, 2007.",

        "[7]  International Maritime Organization, \"Code of Safety for Nuclear "
        "Merchant Ships,\" Resolution A.491(XII), IMO, London, 1981.",

        "[8]  International Atomic Energy Agency, \"Safety of Nuclear Power Plants: "
        "Design,\" Specific Safety Requirements No. SSR-2/1 (Rev. 1), IAEA Safety "
        "Standards Series, Vienna, 2016.",

        "[9]  American Society of Mechanical Engineers, \"ASME Boiler and Pressure "
        "Vessel Code, Section III, Division 5: High Temperature Reactors,\" "
        "ASME, New York, 2021.",

        "[10] Det Norske Veritas Germanischer Lloyd, \"Rules for Classification of "
        "Ships,\" Part 4, Chapter 10: Nuclear Propulsion, DNV GL, 2020.",

        # 11-15: Textbooks
        "[11] Duderstadt, J.J. and Hamilton, L.J., \"Nuclear Reactor Analysis,\" "
        "John Wiley & Sons, New York, 1976.",

        "[12] Lamarsh, J.R. and Baratta, A.J., \"Introduction to Nuclear "
        "Engineering,\" 4th ed., Pearson, Upper Saddle River, NJ, 2017.",

        "[13] Todreas, N.E. and Kazimi, M.S., \"Nuclear Systems, Volume I: "
        "Thermal Hydraulic Fundamentals,\" 2nd ed., CRC Press, Boca Raton, FL, "
        "2012.",

        "[14] El-Wakil, M.M., \"Nuclear Heat Transport,\" International Textbook "
        "Company, Scranton, PA, 1971.",

        "[15] Kern, D.Q., \"Process Heat Transfer,\" McGraw-Hill, New York, 1950.",

        # 16-20: Additional References
        "[16] Incropera, F.P., DeWitt, D.P., Bergman, T.L., and Lavine, A.S., "
        "\"Fundamentals of Heat and Mass Transfer,\" 7th ed., John Wiley & Sons, "
        "2011.",

        "[17] American Nuclear Society, \"Decay Heat Power in Light Water "
        "Reactors,\" ANSI/ANS-5.1-2014, ANS, La Grange Park, IL, 2014.",

        "[18] Shultis, J.K. and Faw, R.E., \"Radiation Shielding,\" American "
        "Nuclear Society, La Grange Park, IL, 2000.",

        "[19] Stacey, W.M., \"Nuclear Reactor Physics,\" 2nd ed., Wiley-VCH, "
        "Weinheim, 2007.",

        "[20] Idaho National Laboratory, \"Fluoride Salt-Cooled High-Temperature "
        "Reactor (FHR) Materials, Fuels, and Components White Paper,\" "
        "INL/EXT-10-18297, Idaho Falls, ID, 2010.",

        # 21-25: Additional MSR, sCO2, Ship, Hastelloy references
        "[21] Jeong, Y.S., Park, S.H., and Bang, I.C., \"Thermal-Hydraulic "
        "Analysis of Molten Salt Reactors for Marine Applications,\" Nuclear "
        "Engineering and Technology, Vol. 54, No. 5, pp. 1803-1815, 2022.",

        "[22] International Maritime Organization, \"2023 IMO Strategy on "
        "Reduction of GHG Emissions from Ships,\" Resolution MEPC.377(80), "
        "IMO, London, 2023.",

        "[23] Dostal, V., Driscoll, M.J., and Hejzlar, P., \"A Supercritical "
        "Carbon Dioxide Cycle for Next Generation Nuclear Reactors,\" "
        "MIT-ANP-TR-100, Massachusetts Institute of Technology, 2004.",

        "[24] Ren, W. and Swindeman, R., \"A Review of Alloy 800H for "
        "Applications in the Gen IV Nuclear Energy Systems,\" ASME Pressure "
        "Vessels and Piping Conference, PVP2009-77033, 2009. "
        "(Hastelloy-N data referenced from ORNL-TM-3866, McCoy, H.E., 1978)",

        "[25] Gat, U. and Engel, J.R., \"The Molten Salt Reactor Experiment "
        "(MSRE) as a Test Facility: Final Report,\" ORNL-TM-5190, "
        "Oak Ridge National Laboratory, 1975.",

        "[26] Forsberg, C.W., \"Molten-Salt-Reactor Technology Gaps,\" "
        "Proceedings of the International Congress on Advances in Nuclear Power "
        "Plants (ICAPP 2006), Reno, NV, 2006.",

        "[27] LeBlanc, D., \"Molten Salt Reactors: A New Beginning for an Old "
        "Idea,\" Nuclear Engineering and Design, Vol. 240, No. 6, pp. 1644-1656, "
        "2010.",

        "[28] Ahn, Y., Bae, S.J., Kim, M., Cho, S.K., Baik, S., Lee, J.I., "
        "and Cha, J.E., \"Review of Supercritical CO2 Power Cycle Technology "
        "and Current Status of Research and Development,\" Nuclear Engineering "
        "and Technology, Vol. 47, No. 6, pp. 647-661, 2015.",
    ]

    pdf.set_font("AppleSD", "", 9)
    for ref in references:
        if pdf.get_y() > 297 - 25 - 10:
            pdf.add_page()

        pdf.set_x(25)
        pdf.multi_cell(w=165, h=5.5, text=ref)
        pdf.ln(2.5)

    # ===================================================================
    # 부록 A: 설계 매개변수 종합표
    # ===================================================================
    pdf.add_page()
    pdf._current_chapter = "부록"
    pdf._toc_entries.append((0, None, "부록 A: 설계 매개변수 종합표", pdf.page_no()))

    # Title
    pdf.set_font("AppleSD", "B", 18)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(w=0, h=10, text="부록 A: 설계 매개변수 종합표")
    pdf.ln(4)

    pdf.set_draw_color(26, 54, 93)
    pdf.set_line_width(1.0)
    y = pdf.get_y()
    pdf.line(25, y, 65, y)
    pdf.set_line_width(0.3)
    pdf.line(67, y, 190, y)
    pdf.ln(10)

    pdf.set_text_color(45, 55, 72)

    pdf.body_text(
        "본 부록은 40 MWth 해양용 용융염 원자로의 모든 핵심 설계 매개변수를 "
        "단일 참조 표로 종합한 것이다. 각 매개변수의 출처 장(chapter)을 표기하여 "
        "상세 내용의 추적을 용이하게 하였다."
    )

    # --- 원자로 설계 기준 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["열출력", "40", "MWth", "Ch.1"],
            ["전기출력", "~16", "MWe", "Ch.1"],
            ["열효율 (sCO₂ Brayton)", "40", "%", "Ch.8"],
            ["설계 수명", "20", "년", "Ch.1"],
            ["용량계수", "85", "%", "Ch.1"],
            ["운전 압력", "0.2", "MPa", "Ch.1"],
            ["노심 입구 온도", "600", "도C", "Ch.1"],
            ["노심 출구 온도", "700", "도C", "Ch.1"],
            ["노심 평균 온도", "650", "도C", "Ch.1"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.1 원자로 설계 기준"
    )

    # --- 노심 기하학 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["노심 직경", "~1.17", "m", "Ch.2"],
            ["노심 높이", "~1.41", "m", "Ch.2"],
            ["노심 체적", "~1.82", "m³", "Ch.2"],
            ["H/D 비", "1.2", "-", "Ch.2"],
            ["흑연 체적분율", "77", "%", "Ch.2"],
            ["연료염 체적분율", "23", "%", "Ch.2"],
            ["채널 직경", "25", "mm", "Ch.2"],
            ["채널 피치", "50", "mm", "Ch.2"],
            ["채널 수", "~400~500", "개", "Ch.2"],
            ["노심 출력밀도", "~22", "MW/m³", "Ch.2"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.2 노심 기하학"
    )

    # --- 연료염 물성 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["연료염 조성", "LiF-BeF₂-UF₄", "-", "Ch.1"],
            ["LiF 몰분율", "64.5", "%", "Ch.1"],
            ["BeF₂ 몰분율", "30.5", "%", "Ch.1"],
            ["UF₄ 몰분율", "5.0", "%", "Ch.1"],
            ["⁷Li 농축도", "99.995", "%", "Ch.1"],
            ["밀도 (650도C)", "~2,095", "kg/m³", "Ch.3"],
            ["점도 (650도C)", "~8.6", "mPa-s", "Ch.3"],
            ["비열", "2,386", "J/(kg-K)", "Ch.3"],
            ["열전도도", "1.1", "W/(m-K)", "Ch.3"],
            ["융점", "459", "도C", "Ch.1"],
            ["비점", "~1,400", "도C", "Ch.1"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.3 연료염 물성 (FLiBe + UF₄)"
    )

    # --- 핵특성 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["U-235 농축도", "~12", "% (HALEU)", "Ch.2"],
            ["우라늄 장전량", "~120~150", "kg", "Ch.2"],
            ["U-235 질량", "~15~18", "kg", "Ch.2"],
            ["핵분열당 에너지", "200", "MeV", "Ch.2"],
            ["핵분열당 중성자 (nu)", "2.43", "-", "Ch.2"],
            ["지발 중성자 분율 (beta)", "0.0065", "-", "Ch.2"],
            ["즉발 중성자 수명", "4.0 x 10⁻⁴", "s", "Ch.2"],
            ["alpha_fuel (연료 온도)", "< 0 (음)", "pcm/K", "Ch.2"],
            ["alpha_graphite (흑연 온도)", "< 0 (음)", "pcm/K", "Ch.2"],
            ["alpha_density (밀도)", "< 0 (음)", "pcm/(kg/m³)", "Ch.2"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.4 핵특성"
    )

    # --- 열수력 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["질량유량", "~167", "kg/s", "Ch.3"],
            ["체적유량", "~0.080", "m³/s", "Ch.3"],
            ["채널 내 유속", "~1.0~1.5", "m/s", "Ch.3"],
            ["노심 체류시간", "~8~12", "s", "Ch.3"],
            ["레이놀즈 수", "~600~1,200", "-", "Ch.3"],
            ["프란틀 수", "~18~20", "-", "Ch.3"],
            ["노심 압력손실", "~50~80", "kPa", "Ch.3"],
            ["펌프 동력", "~5~10", "kW", "Ch.3"],
            ["첨두 연료 온도 (핫채널)", "~760~780", "도C", "Ch.3"],
            ["첨두 흑연 온도", "~830", "도C", "Ch.3"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.5 열수력 매개변수"
    )

    # --- 열교환기 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["형식", "Shell-and-Tube", "-", "Ch.4"],
            ["전열 용량", "40", "MW", "Ch.4"],
            ["LMTD", "~50", "K", "Ch.4"],
            ["총괄 열전달 계수", "~1,800~2,200", "W/(m²K)", "Ch.4"],
            ["전열면적", "~40~50", "m²", "Ch.4"],
            ["튜브 수", "~400~600", "개", "Ch.4"],
            ["튜브 길이", "~3~4", "m", "Ch.4"],
            ["셸 직경", "~0.5~0.6", "m", "Ch.4"],
            ["유효도", "> 95", "%", "Ch.4"],
            ["재질", "Hastelloy-N", "-", "Ch.4"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.6 1차 열교환기"
    )

    # --- 구조 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["용기 재질", "Hastelloy-N", "-", "Ch.5"],
            ["용기 벽두께", "20 (최소)", "mm", "Ch.5"],
            ["용기 내경", "~1.4", "m", "Ch.5"],
            ["용기 높이", "~2.6", "m", "Ch.5"],
            ["설계 압력", "0.2", "MPa", "Ch.5"],
            ["허용 응력", "55", "MPa", "Ch.5"],
            ["크리프 파단 (700도C, 10khr)", "83", "MPa", "Ch.5"],
            ["기초 볼트 수", "12", "개", "Ch.5"],
            ["볼트 안전율", "> 3.0", "-", "Ch.5"],
            ["자유 표면 GG'", "0.003", "mm", "Ch.5"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.7 구조 설계"
    )

    # --- 차폐 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["중성자 선원 강도", "3.034 x 10¹⁸", "n/s", "Ch.7"],
            ["총 감마선원 강도", "1.869 x 10¹⁹", "ph/s", "Ch.7"],
            ["Hastelloy-N (용기벽)", "2", "cm", "Ch.7"],
            ["강재", "25", "cm", "Ch.7"],
            ["B₄C", "15", "cm", "Ch.7"],
            ["중량 콘크리트 (Baryte)", "120", "cm", "Ch.7"],
            ["총 차폐 두께", "162", "cm", "Ch.7"],
            ["차폐 외경", "~5.0", "m", "Ch.7"],
            ["차폐 높이", "~5.8", "m", "Ch.7"],
            ["구획경계 선량률 (5m)", "6.82", "uSv/hr", "Ch.7"],
            ["거주구역 연간 선량 (15m)", "0.30", "mSv/yr", "Ch.7"],
            ["차폐 추정 중량", "~250", "톤", "Ch.7"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.8 차폐 설계"
    )

    # --- 안전 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["ULOF 첨두 연료 온도", "비등점 이하", "-", "Ch.6"],
            ["UTOP 반응도 삽입", "50", "pcm", "Ch.6"],
            ["SBO 자연순환 능력", "붕괴열 제거 충분", "-", "Ch.6"],
            ["비상 배수 탱크 k_eff", "< 0.95", "-", "Ch.6"],
            ["직업피폭 한계", "20", "mSv/yr", "ICRP 103"],
            ["공중피폭 한계", "1", "mSv/yr", "ICRP 103"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.9 안전 매개변수"
    )

    # --- 선박 통합 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["대상 선박", "6,000 TEU Panamax", "-", "Ch.8"],
            ["배수량", "80,000", "톤", "Ch.8"],
            ["서비스 속력", "18", "노트", "Ch.8"],
            ["기관실 치수", "25 x 20 x 12", "m", "Ch.8"],
            ["원자로 시스템 총 중량", "~386", "톤", "Ch.8"],
            ["기관실 체적 점유율", "~5.0", "%", "Ch.8"],
            ["동력변환", "sCO₂ Brayton (40%)", "-", "Ch.8"],
            ["추진 출력", "14", "MW (축출력)", "Ch.8"],
            ["추진 방식", "전기 추진 (PMSM)", "-", "Ch.8"],
            ["20년 연료비 절감", "2.4~3.8", "억 USD", "Ch.8"],
            ["연간 CO₂ 절감", "~112,000", "톤/yr", "Ch.8"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.10 선박 통합"
    )

    # --- Hastelloy-N 물성 ---
    pdf.add_table(
        headers=["매개변수", "값", "단위", "출처"],
        rows=[
            ["밀도", "8,860", "kg/m³", "Ch.1"],
            ["융점", "1,372", "도C", "Ch.1"],
            ["열팽창계수 (21~316도C)", "12.3 x 10⁻⁶", "1/K", "Ch.1"],
            ["탄성계수 (실온)", "219", "GPa", "Ch.5"],
            ["포아송 비", "0.32", "-", "Ch.5"],
            ["열전도도 (600도C)", "~21.7", "W/(m-K)", "Ch.1"],
            ["허용 응력 (ASME)", "55", "MPa", "Ch.5"],
            ["크리프 파단 (700도C, 10khr)", "83", "MPa", "Ch.5"],
            ["FLiBe 내 부식률", "~25", "um/yr", "Ch.1"],
            ["최대 사용 온도", "704", "도C", "Ch.1"],
        ],
        col_widths=[55, 30, 40, 40],
        title="표 A.11 Hastelloy-N 주요 물성"
    )

    pdf.add_note(
        "본 부록의 매개변수 값은 개념설계 수준의 대표값이며, 일부 범위로 "
        "표시된 값은 설계 반복(iteration) 과정에서의 변동 범위를 나타낸다. "
        "상세 설계에서는 보다 정밀한 해석에 의해 최종 확정되어야 한다."
    )
