import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

# NUCLEAR LAB - Materi Bab I, II, III Fisika Nuklir
# Materi dirangkum dan disusun berdasarkan Buku Ajar Fisika Nuklir, Choirul Anam, Much. Azam, K. Sofjan Firdausi (2007).

st.set_page_config(
    page_title="NUCLEAR LAB | Fisika Inti",
    page_icon="N",
    layout="wide"
)

# ------------------------- STYLE -----------------------------
st.markdown("""
<style>
.stApp { background: #071526; color: #E8F4FF; }
h1, h2, h3 { color: #42D9FF; }
p, li, label { color: #D5E5F2; }
[data-testid="stSidebar"] { background: #0B1D31; }
.stButton > button {
    background: #087EAA; color: White; border: 0;
    border-radius: 9px; font-weight: 650;
}
.stButton > button:hover { background: #12B8DF; color: #061523; }
.kartu {
    background: #10263D; border: 1px solid #1D5875;
    border-radius: 14px; padding: 18px; margin: 8px 0;
}
.kecil { color: #A9C4D8; font-size: 0.9rem; }

/* Tema gelap untuk kolom masukan agar teks terbaca */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #10263D !important;
    color: #F3F8FC !important;
    border: 1px solid #3B6683 !important;
    caret-color: #42D9FF !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder,
[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder {
    color: #A9C4D8 !important;
    opacity: 1 !important;
}
[data-baseweb="select"] > div {
    background-color: #10263D !important;
    color: #F3F8FC !important;
    border-color: #3B6683 !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #F3F8FC !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #10263D !important;
    color: #F3F8FC !important;
}
[role="option"]:hover {
    background-color: #1D5875 !important;
}
.stRadio label, .stCheckbox label,
.stSlider label, .stSelectbox label,
.stNumberInput label, .stTextInput label,
.stTextArea label, .stMultiSelect label {
    color: #D5E5F2 !important;
}
.stRadio [data-testid="stMarkdownContainer"] p,
.stCheckbox [data-testid="stMarkdownContainer"] p {
    color: #E8F4FF !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #42D9FF !important;
}
.stForm, [data-testid="stForm"] {
    background-color: #0B1D31 !important;
    border: 1px solid #24445D !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------- HELPER VISUAL -------------------------
def setup_ax(ax, title=None):
    ax.set_facecolor("#10263D")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#55758C")
    ax.grid(alpha=0.18)
    if title:
        ax.set_title(title, color="white", pad=10)

def atom_model(model):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor("#10263D")
    ax.set_facecolor("#10263D")
    if model == "Dalton":
        ax.add_patch(Circle((0, 0), 0.85, color="#42D9FF"))
        ax.text(0, 0, "Atom", color="#071526", ha="center", va="center",
                fontsize=13, weight="bold")
    elif model == "Thomson":
        ax.add_patch(Circle((0, 0), 1, color="#F5A623", alpha=.9))
        for x, y in [(-.45,.35),(.35,.45),(-.3,-.4),(.48,-.25),(0,0)]:
            ax.add_patch(Circle((x,y), .12, color="#071526"))
    elif model == "Rutherford":
        ax.add_patch(Circle((0,0),.15,color="#FF5555"))
        for angle in (0, 60, 120):
            ax.add_patch(Ellipse((0,0),2.1,.8,angle=angle,fill=False,
                                 edgecolor="#42D9FF",lw=1.8))
        ax.scatter([.95,-.65,.0],[.2,-.35,.9],c="white",s=35)
    elif model == "Bohr":
        ax.add_patch(Circle((0,0),.18,color="#FF5555"))
        for r in (.55,1.05):
            ax.add_patch(Circle((0,0),r,fill=False,color="#42D9FF",lw=1.8))
        ax.scatter([.55,-1.05],[0,0],c="white",s=45)
    else:
        ax.add_patch(Circle((0,0),.16,color="#FF5555"))
        rng = np.random.default_rng(12)
        x, y = rng.normal(0,.48,150), rng.normal(0,.48,150)
        ax.scatter(x,y,s=9,c="#42D9FF",alpha=.22)
        ax.scatter(x[::8],y[::8],s=12,c="white",alpha=.55)
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.3,1.3)
    ax.set_aspect("equal"); ax.axis("off")
    return fig

def nucleus_diagram(Z, N, seed=10, title="Ilustrasi inti atom"):
    fig, ax = plt.subplots(figsize=(5.8,4.0))
    fig.patch.set_facecolor("#10263D")
    ax.set_facecolor("#10263D")
    rng = np.random.default_rng(seed)
    count = Z + N
    if count > 0:
        angles = rng.uniform(0, 2*np.pi, count)
        radii = .88*np.sqrt(rng.uniform(0,1,count))
        points = list(zip(radii*np.cos(angles), radii*np.sin(angles)))
        for i,(x,y) in enumerate(points):
            color = "#FF6262" if i < Z else "#45A7F5"
            ax.add_patch(Circle((x,y), .105, color=color,
                                ec="white", lw=.45))
    ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.2,1.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color="white")
    return fig

def isotope_diagram():
    fig, axes = plt.subplots(1,3,figsize=(9,3.1))
    fig.patch.set_facecolor("#10263D")
    for ax,(Z,N,label) in zip(axes,[(6,6,"Karbon-12"),(6,7,"Karbon-13"),(6,8,"Karbon-14")]):
        ax.set_facecolor("#10263D")
        rng=np.random.default_rng(Z+N)
        for i in range(Z+N):
            a=rng.uniform(0,2*np.pi); r=.8*np.sqrt(rng.uniform())
            ax.add_patch(Circle((r*np.cos(a),r*np.sin(a)),.105,
                         color="#FF6262" if i<Z else "#45A7F5"))
        ax.set_title(label,color="white",fontsize=10)
        ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15)
        ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    return fig

def radius_plot():
    A=np.arange(1,241)
    R=1.2*A**(1/3)
    fig,ax=plt.subplots(figsize=(7,3.5)); setup_ax(ax)
    ax.plot(A,R,color="#42D9FF",lw=2.5)
    ax.set_xlabel("Nomor massa, A",color="white")
    ax.set_ylabel("Jari-jari, R (fm)",color="white")
    return fig

def mass_spectrometer_diagram():
    fig,ax=plt.subplots(figsize=(7,3.4))
    fig.patch.set_facecolor("#10263D"); ax.set_facecolor("#10263D")
    ax.add_patch(Rectangle((.2,.8),.7,.45,color="#F5A623"))
    ax.text(.55,1.02,"Sumber ion",ha="center",color="#071526")
    ax.add_patch(Rectangle((1.35,.65),.12,.75,color="#DCEAF4"))
    ax.text(1.4,1.55,"Celah",ha="center",color="white")
    ax.add_patch(Rectangle((2.1,.35),2.3,1.4,fill=False,ec="#42D9FF",lw=2))
    ax.text(3.25,1.55,"Medan magnet",ha="center",color="white")
    theta=np.linspace(-np.pi/2,np.pi/2,100)
    ax.plot(2.3+1.0*np.cos(theta),1.05+0.7*np.sin(theta),
            color="#F5A623",lw=2)
    ax.add_patch(Rectangle((4.75,.4),.12,1.3,color="#DCEAF4"))
    ax.text(4.8,1.8,"Detektor",ha="center",color="white")
    ax.annotate("",xy=(2.25,1.05),xytext=(1.5,1.05),
                arrowprops=dict(arrowstyle="->",color="white",lw=2))
    ax.set_xlim(0,5.5); ax.set_ylim(0,2.1); ax.axis("off")
    return fig

def binding_curve():
    A=np.arange(2,241)
    # Kurva skematis: puncak di sekitar besi, bukan data isotop individual.
    B=8.8*(1-np.exp(-A/18))*np.exp(-np.maximum(A-56,0)/700)
    fig,ax=plt.subplots(figsize=(7,3.8)); setup_ax(ax)
    ax.plot(A,B,color="#42D9FF",lw=2.6)
    ax.scatter([56],[8.8*(1-np.exp(-56/18))],
               color="#F5A623",s=55,label="Daerah sekitar Fe-56")
    ax.set_xlabel("Nomor massa, A",color="white")
    ax.set_ylabel("Energi ikat per nukleon (MeV)",color="white")
    ax.legend(facecolor="#10263D",labelcolor="white")
    return fig

def force_range_plot():
    r=np.linspace(.15,3,400)
    # Skema kualitatif: tarikan pada jarak menengah, tolakan kuat pada jarak sangat kecil.
    V=38*np.exp(-((r-.75)/.42)**2)-65*np.exp(-((r-.23)/.12)**2)
    V*=np.exp(-np.maximum(r-1.5,0)*2.5)
    fig,ax=plt.subplots(figsize=(7,3.5)); setup_ax(ax)
    ax.plot(r,V,color="#42D9FF",lw=2.5)
    ax.axhline(0,color="white",lw=.8)
    ax.axvline(1,color="#F5A623",ls="--",label="Sekitar 1 fm")
    ax.set_xlabel("Jarak antar nukleon (fm)",color="white")
    ax.set_ylabel("Potensial (skematis)",color="white")
    ax.legend(facecolor="#10263D",labelcolor="white")
    return fig

def liquid_drop_diagram():
    fig,ax=plt.subplots(figsize=(6,3.7))
    fig.patch.set_facecolor("#10263D"); ax.set_facecolor("#10263D")
    rng=np.random.default_rng(5)
    for i in range(75):
        a=rng.uniform(0,2*np.pi); r=.9*np.sqrt(rng.uniform())
        ax.add_patch(Circle((r*np.cos(a),r*np.sin(a)),.075,
                            color="#45A7F5" if i%2 else "#FF6262"))
    ax.add_patch(Ellipse((0,0),2.35,2.35,fill=False,
                         ec="#F5A623",lw=2,ls="--"))
    ax.text(0,-1.45,"Nukleon bergerak secara kolektif",ha="center",
            color="white",fontsize=10)
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.7,1.35)
    ax.set_aspect("equal"); ax.axis("off")
    return fig

def shell_diagram():
    fig,ax=plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor("#10263D"); ax.set_facecolor("#10263D")
    levels=[1.0,1.8,2.7,3.6]
    labels=["Kulit 1","Kulit 2","Kulit 3","Kulit 4"]
    for i,(y,label) in enumerate(zip(levels,labels)):
        ax.plot([.5,4.7],[y,y],color="#42D9FF",lw=2)
        ax.text(4.85,y,label,color="white",va="center")
    ax.annotate("Pengisian tingkat energi",
                xy=(1.1,3.6),xytext=(1.1,.5),
                color="#F5A623",
                arrowprops=dict(arrowstyle="->",color="#F5A623",lw=2))
    ax.set_xlim(0,6); ax.set_ylim(0,4.3); ax.axis("off")
    return fig


# ============================================================
# FUNGSI TAMBAHAN SIMULASI BAB II
# ============================================================
MH_U, MN_U, C2_MEV, HC_MEVFM = 1.007825, 1.008665, 931.5, 197.3

# nama: (Z, A, massa atom dalam u)
DATA_INTI = {
    "H-2": (1, 2, 2.014102), "H-3": (1, 3, 3.016049), "He-4": (2, 4, 4.002603),
    "Li-6": (3, 6, 6.015123), "C-12": (6, 12, 12.0), "O-16": (8, 16, 15.994915),
    "Ne-20": (10, 20, 19.99244), "Mg-24": (12, 24, 23.985042),
    "Si-28": (14, 28, 27.976927), "Ca-40": (20, 40, 39.962591),
    "Fe-56": (26, 56, 55.934936), "Ni-62": (28, 62, 61.928345),
    "Zr-90": (40, 90, 89.904704), "Kr-92": (36, 92, 91.926156),
    "Te-126": (52, 126, 125.903322), "Ba-138": (56, 138, 137.905247),
    "Ba-141": (56, 141, 140.914411), "Au-197": (79, 197, 196.966569),
    "Pb-208": (82, 208, 207.976652), "U-235": (92, 235, 235.04393),
    "U-238": (92, 238, 238.050788),
}

REAKSI = {
    "H-2 + H-3 → He-4 + n (fusi)": (["H-2", "H-3"], ["He-4", "n"]),
    "C-12 + C-12 → Mg-24 (fusi)": (["C-12", "C-12"], ["Mg-24"]),
    "U-235 + n → Ba-141 + Kr-92 + 3n (fisi)": (
        ["U-235", "n"], ["Ba-141", "Kr-92", "n", "n", "n"]
    ),
}


def energi_ikat_inti(Z, A, m):
    """Energi ikat berdasarkan massa atom, dalam MeV."""
    return (Z * MH_U + (A - Z) * MN_U - m) * C2_MEV


def grafik_ba_reaksi(nama_reaksi, inti_diubah, dm):
    pereaksi, hasil = REAKSI[nama_reaksi]

    def massa(nama):
        if nama == "n":
            return MN_U
        m = DATA_INTI[nama][2]
        return m + dm if nama == inti_diubah else m

    urut = sorted(DATA_INTI.items(), key=lambda kv: kv[1][1])
    nomor_massa = [v[1] for _, v in urut]
    ba = [
        energi_ikat_inti(v[0], v[1], massa(n)) / v[1]
        for n, v in urut
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor("#10263D")
    setup_ax(ax)
    ax.plot(nomor_massa, ba, color="#55758C", lw=1)
    ax.scatter(nomor_massa, ba, color="#42D9FF", zorder=3)

    for nama, warna in (
        [(n, "#F5A623") for n in pereaksi]
        + [(n, "white") for n in hasil]
    ):
        if nama != "n":
            Z, a, _ = DATA_INTI[nama]
            ax.scatter(
                [a], [energi_ikat_inti(Z, a, massa(nama)) / a],
                s=230, facecolors="none", edgecolors=warna,
                linewidths=2.5, zorder=4
            )

    Z, a, _ = DATA_INTI[inti_diubah]
    ba_terpilih = energi_ikat_inti(Z, a, massa(inti_diubah)) / a
    ax.annotate(
        inti_diubah, (a, ba_terpilih), textcoords="offset points",
        xytext=(10, 10), color="white", fontweight="bold"
    )

    Q = (
        sum(massa(n) for n in pereaksi)
        - sum(massa(n) for n in hasil)
    ) * C2_MEV

    ax.set_xlim(0, 250)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Nomor massa, A", color="white")
    ax.set_ylabel("Energi ikat per nukleon (MeV)", color="white")
    fig.tight_layout()
    return fig, Q, ba_terpilih


def grafik_jangkauan_pendek(a, k, r0, A0):
    A = np.linspace(0, 250, 251)
    data_A = np.array([v[1] for v in DATA_INTI.values()])
    data_B = np.array([energi_ikat_inti(*v) for v in DATA_INTI.values()])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("#10263D")
    setup_ax(ax1)
    setup_ax(ax2)

    ax1.plot(A, a * A, color="#42D9FF", lw=2.2,
             label="jangkauan pendek: E = a·A")
    ax1.plot(A, k * A**2, color="#F5A623", lw=2.2,
             label="jangkauan panjang: E = k·A²")
    ax1.scatter(data_A, data_B, color="white", s=16, zorder=3,
                label="data massa inti")
    ax1.axvline(A0, ls="--", color="#A9C4D8")
    ax1.set_ylim(0, 2500)
    ax1.set_xlabel("Nomor massa, A", color="white")
    ax1.set_ylabel("Energi ikat total (MeV)", color="white")

    ax2.plot(A, np.full_like(A, a), color="#42D9FF", lw=2.2,
             label="E/A = a")
    ax2.plot(A, k * A, color="#F5A623", lw=2.2,
             label="E/A = k·A")
    ax2.scatter(data_A, data_B / data_A, color="white", s=16, zorder=3)
    ax2.axvline(A0, ls="--", color="#A9C4D8")
    ax2.set_ylim(0, 12)
    ax2.set_xlabel("Nomor massa, A", color="white")
    ax2.set_ylabel("E / A (MeV)", color="white")

    for ax in (ax1, ax2):
        ax.legend(facecolor="#10263D", labelcolor="white",
                  fontsize=8, loc="upper left")

    fig.tight_layout()
    rho = 3 / (4 * np.pi * r0**3)
    return fig, rho


def grafik_yukawa(E, d):
    x = HC_MEVFM / E
    dt = 6.582e-22 / E
    Es = np.linspace(10, 1000, 300)

    fig, (ax_il, ax_g) = plt.subplots(
        2, 1, figsize=(7.5, 6.5),
        gridspec_kw={"height_ratios": [1, 1.8]}
    )
    fig.patch.set_facecolor("#10263D")
    ax_il.set_facecolor("#10263D")

    xp, xn = -d / 2, d / 2
    ax_il.add_patch(Circle((xp, 0), x, fill=False, ls="--",
                           ec="#42D9FF", lw=1.8))
    ax_il.add_patch(Circle((xp, 0), 0.12, color="#FF6262"))
    ax_il.add_patch(Circle((xn, 0), 0.12, color="#45A7F5"))
    ax_il.plot(
        [xp + 0.12, xp + min(d, x) / 2 + 0.12],
        [0, 0], ".", color="white", ms=8
    )
    ax_il.set_xlim(-5, 5)
    ax_il.set_ylim(-2.5, 2.5)
    ax_il.set_aspect("equal")
    ax_il.axis("off")

    setup_ax(ax_g)
    ax_g.plot(Es, HC_MEVFM / Es, color="#42D9FF", lw=2.5)
    ax_g.scatter(
        [140, 200], [HC_MEVFM / 140, HC_MEVFM / 200],
        color=["#FF6262", "#45A7F5"], zorder=3
    )
    ax_g.annotate(
        "meson pi", (140, HC_MEVFM / 140), xytext=(8, 8),
        textcoords="offset points", color="white"
    )
    ax_g.annotate(
        "taksiran buku", (200, HC_MEVFM / 200), xytext=(8, -14),
        textcoords="offset points", color="white"
    )
    ax_g.scatter([E], [x], s=90, color="#F5A623", zorder=4)
    ax_g.set_xlim(0, 1000)
    ax_g.set_ylim(0, 6)
    ax_g.set_xlabel("Energi diam partikel, mc² (MeV)", color="white")
    ax_g.set_ylabel("Jangkauan, x (fm)", color="white")
    fig.tight_layout()
    return fig, x, dt, x >= d


# ------------------------- HEADER ----------------------------
st.title("NUCLEAR LAB")
st.subheader("Laboratorium Virtual Fisika Inti")
st.write(
    "Media pembelajaran interaktif Fisika Inti. "
    "Materi Bab I-III disusun berdasarkan buku ajar Fisika Nuklir "
    "karya Choirul Anam, Much. Azam, dan K. Sofjan Firdausi (2007)."
)
st.divider()

menu = st.sidebar.radio(
    "MENU PEMBELAJARAN",
    [
        "Beranda",
        "I. Struktur Materi",
        "I. Sifat Inti",
        "II. Energi Ikat",
        "II. Gaya Inti",
        "III. Model Tetes Cairan",
        "III. Model Kulit",
        "Simulasi",
        "Latihan Soal",
        "Evaluasi"
    ]
)

# ============================================================
# BERANDA
# ============================================================
if menu == "Beranda":
    st.header("Selamat Datang di Nuclear Lab")
    st.markdown("""
    <div class="kartu">
    <h3>Belajar Fisika Inti melalui Laboratorium Virtual</h3>
    Fisika inti mempelajari struktur, sifat, energi, gaya, dan model
    yang digunakan untuk menjelaskan inti atom. Pada laboratorium ini,
    materi disajikan melalui uraian konsep, persamaan, ilustrasi,
    simulasi, dan latihan.
    </div>
    """,unsafe_allow_html=True)

    a,b,c=st.columns(3)
    a.metric("Bab Pembelajaran","3 Bab")
    b.metric("Subbab Materi","6 Subbab")
    c.metric("Fitur Pembelajaran","Simulasi & Latihan Soal")

    st.subheader("Peta Materi")
    st.markdown("""
    **Bab I - Struktur dan Sifat Inti**
    - Struktur materi dan perkembangan model atom.
    - Partikel penyusun inti, lambang nuklida, isotop, isobar, isoton.
    - Massa inti, spektrometer massa, jari-jari inti.

    **Bab II - Energi Ikat dan Gaya Inti**
    - Massa defek, energi ikat, energi ikat per nukleon.
    - Sifat gaya inti, jangkauan gaya, dan model pertukaran Yukawa.

    **Bab III - Model Inti**
    - Model tetes cairan dan koreksi persamaan semiempiris.
    - Model kulit, bilangan ajaib, tingkat energi, dan spin-orbit.
    """)
    st.info("Pilih salah satu materi melalui menu di sebelah kiri.")

# ============================================================
# BAB I A. STRUKTUR MATERI
# ============================================================
elif menu == "I. Struktur Materi":
    st.header("I. STRUKTUR DAN SIFAT INTI")
    st.subheader("A. Struktur Materi")

    st.markdown("""
    ### 1. Pengertian dan gagasan tentang materi

    Materi merupakan segala sesuatu yang memiliki massa dan menempati ruang.
    Sejak zaman Yunani kuno, muncul pertanyaan apakah materi dapat dibagi
    terus-menerus atau tersusun atas bagian-bagian terkecil.

    Democritus (sekitar 450 SM) mengusulkan bahwa materi tersusun dari
    partikel sangat kecil yang tidak dapat dibagi lagi, yang disebut atom.
    Gagasan ini pada awalnya masih berupa pemikiran filosofis dan belum
    dapat dibuktikan melalui eksperimen.

    Pada tahun 1803, John Dalton mengembangkan teori atom berdasarkan
    hasil pengamatan kimia. Dalton memandang atom sebagai partikel dasar
    suatu unsur. Atom-atom dari unsur yang sama dianggap serupa dan
    atom-atom dari unsur berbeda dapat bergabung membentuk molekul.
    """)

    st.markdown("### 2. Perkembangan model atom")
    st.write(
        "Penemuan partikel subatom menyebabkan model atom mengalami "
        "perubahan. Pilih model berikut untuk melihat ilustrasi dan penjelasannya."
    )

    model_data = {
        "Dalton": (
            "Atom dipandang sebagai bola pejal yang sangat kecil. "
            "Model ini membantu menjelaskan hukum-hukum dasar reaksi kimia, "
            "tetapi belum menjelaskan partikel penyusun atom."
        ),
        "Thomson": (
            "Setelah elektron ditemukan oleh J. J. Thomson, atom tidak lagi "
            "dipandang sebagai partikel yang tidak dapat dibagi. Thomson "
            "menggambarkan atom sebagai bola bermuatan positif yang di dalamnya "
            "tersebar elektron bermuatan negatif. Jumlah muatan positif dan "
            "negatif membuat atom bersifat netral."
        ),
        "Rutherford": (
            "Rutherford menembakkan partikel alfa pada lempeng emas tipis. "
            "Sebagian besar partikel alfa menembus lurus, sebagian dibelokkan, "
            "dan sebagian kecil dipantulkan. Hasil ini menunjukkan bahwa atom "
            "sebagian besar berupa ruang kosong dan muatan positif serta "
            "sebagian besar massa terkumpul pada inti yang kecil."
        ),
        "Bohr": (
            "Bohr menyempurnakan model Rutherford. Elektron hanya dapat berada "
            "pada lintasan stasioner tertentu tanpa memancarkan energi. "
            "Elektron menyerap energi ketika berpindah ke tingkat lebih tinggi "
            "dan melepaskan energi ketika berpindah ke tingkat lebih rendah."
        ),
        "Mekanika Kuantum": (
            "Gambaran modern tidak menetapkan elektron bergerak pada lintasan "
            "planet yang pasti. Keadaan elektron dijelaskan melalui orbital, "
            "yaitu daerah peluang ditemukannya elektron."
        )
    }

    pilihan_model=st.selectbox("Pilih model atom",list(model_data.keys()))
    x,y=st.columns([1,1])
    with x:
        st.markdown(f'<div class="kartu"><h3>Model {pilihan_model}</h3>'
                    f'{model_data[pilihan_model]}</div>',unsafe_allow_html=True)
    with y:
        st.pyplot(atom_model(pilihan_model))

    st.markdown("### 3. Mengapa penemuan inti atom penting?")
    st.write("""
    Percobaan Rutherford memperlihatkan bahwa massa atom tidak tersebar
    merata. Sebagian besar massa terkonsentrasi pada inti yang ukurannya
    sangat kecil dibandingkan ukuran atom. Pemahaman ini menjadi dasar
    berkembangnya fisika inti.
    """)
    st.info(
        "Catatan: model atom menjelaskan susunan atom secara keseluruhan, "
        "sedangkan pembahasan berikutnya berfokus pada struktur dan sifat inti."
    )

# ============================================================
# BAB I B. SIFAT INTI
# ============================================================
elif menu == "I. Sifat Inti":
    st.header("I. STRUKTUR DAN SIFAT INTI")
    st.subheader("B. Sifat Inti")

    st.markdown("""
    ### 1. Inti atom

    Inti atom atau nuklida merupakan bagian pusat atom. Ukurannya sangat
    kecil dibandingkan diameter atom, tetapi mengandung hampir seluruh
    massa atom. Inti tersusun atas proton dan neutron yang secara bersama
    disebut nukleon.

    Proton bermuatan positif, sedangkan neutron tidak bermuatan.
    Elektron berada di luar inti. Pada atom netral, jumlah elektron sama
    dengan jumlah proton.
    """)

    st.pyplot(nucleus_diagram(6,6,seed=15,title="Inti atom: proton dan neutron"))
    st.caption("Merah = proton; biru = neutron. Susunan partikel merupakan ilustrasi skematis.")

    st.markdown("### 2. Lambang nuklida")
    st.latex(r"{}^{A}_{Z}X")
    st.write("""
    - X adalah lambang unsur.
    - Z adalah nomor atom, yaitu jumlah proton.
    - A adalah nomor massa, yaitu jumlah proton dan neutron.
    - N adalah jumlah neutron.
    """)
    st.latex(r"A=Z+N \qquad N=A-Z")

    st.markdown("**Contoh: karbon-12**")
    st.latex(r"{}^{12}_{6}\mathrm{C}")
    st.write("Z = 6, A = 12, sehingga N = 12 - 6 = 6 neutron.")

    st.markdown("### 3. Isotop, isobar, dan isoton")
    tab1,tab2,tab3=st.tabs(["Isotop","Isobar","Isoton"])
    with tab1:
        st.write("""
        Isotop adalah nuklida dengan nomor atom sama, tetapi nomor massa
        berbeda. Artinya, jumlah proton sama dan jumlah neutron berbeda.
        Contohnya karbon-12, karbon-13, dan karbon-14.
        """)
        st.pyplot(isotope_diagram())
        st.caption("Ketiga isotop karbon memiliki 6 proton, sedangkan neutronnya berturut-turut 6, 7, dan 8.")
    with tab2:
        st.write("""
        Isobar adalah nuklida dengan nomor massa sama, tetapi nomor atom
        berbeda. Contoh: hidrogen-3 dan helium-3 sama-sama memiliki A = 3,
        tetapi nomor atomnya berbeda.
        """)
        st.latex(r"{}^{3}_{1}\mathrm{H}\quad \text{dan}\quad {}^{3}_{2}\mathrm{He}")
    with tab3:
        st.write("""
        Isoton adalah nuklida yang memiliki jumlah neutron sama,
        tetapi nomor atom berbeda. Contoh: hidrogen-3 memiliki 2 neutron,
        sedangkan helium-4 juga memiliki 2 neutron.
        """)
        st.latex(r"{}^{3}_{1}\mathrm{H}\quad \text{dan}\quad {}^{4}_{2}\mathrm{He}")

    st.markdown("### 4. Massa inti dan satuan massa atom")
    st.write("""
    Massa inti sangat kecil sehingga satuan kilogram kurang praktis
    untuk perhitungan tingkat atom. Karena itu digunakan satuan massa
    atom terpadu, yaitu u. Satu satuan massa atom didefinisikan sebagai
    seperdua belas massa atom karbon-12.
    """)
    st.latex(r"1\,u \approx 1.66\times10^{-27}\,\mathrm{kg}")
    st.latex(r"1\,u\,c^2 \approx 931.5\,\mathrm{MeV}")

    st.markdown("**Massa dan sifat nukleon menurut buku ajar**")
    st.table({
        "Besaran":["Muatan proton","Muatan neutron","Massa proton","Massa neutron","Spin proton dan neutron"],
        "Nilai":["+1,6 Ã— 10â»Â¹â¹ C","0 C","1,007277 u","1,008665 u","1/2"]
    })

    st.markdown("### 5. Pengukuran massa inti: spektrometer massa")
    st.write("""
    Massa partikel bermuatan dapat ditentukan dari geraknya dalam medan
    listrik dan medan magnet. Pada pemilih kecepatan, gaya listrik dan
    gaya magnet saling menyeimbangkan:
    """)
    st.latex(r"qE=qvB \quad \Rightarrow \quad v=\frac{E}{B}")
    st.write("""
    Setelah melewati pemilih kecepatan, ion memasuki medan magnet lain
    dan bergerak melengkung. Gaya Lorentz menjadi gaya sentripetal:
    """)
    st.latex(r"qvB'= \frac{mv^2}{r}")
    st.latex(r"m=\frac{qB'r}{v}")
    st.pyplot(mass_spectrometer_diagram())
    st.caption("Skema konseptual spektrometer massa; lintasan dan komponen disederhanakan.")

    st.markdown("### 6. Jari-jari inti")
    st.write("""
    Rapat nukleon di dalam inti secara umum hampir konstan. Karena
    jumlah nukleon sebanding dengan volume inti, diperoleh hubungan
    jari-jari dengan nomor massa:
    """)
    st.latex(r"R=R_0A^{1/3}")
    st.write("""
    Untuk ukuran distribusi muatan, buku menggunakan nilai pendekatan
    Râ‚€ sekitar 1,2 fm, sedangkan distribusi massa sekitar 1,4 fm.
    Satu femtometer (fm) sama dengan 10â»Â¹âµ meter.
    """)
    st.pyplot(radius_plot())
    st.caption("Grafik hubungan R = 1,2 A^(1/3) fm; kurva menunjukkan kecenderungan, bukan hasil pengukuran tiap isotop.")

    st.markdown("### 7. Cara mengukur jari-jari inti")
    st.write("""
    Salah satu cara adalah menghamburkan partikel bermuatan, misalnya
    partikel alfa, pada inti. Penyimpangan hasil hamburan dari prediksi
    hamburan Rutherford dapat memberikan informasi tentang ukuran inti.
    Cara lain menggunakan hamburan elektron dan pola difraksi.
    """)

# ============================================================
# BAB II A. ENERGI IKAT
# ============================================================
elif menu == "II. Energi Ikat":
    st.header("II. ENERGI IKAT DAN GAYA INTI")
    st.subheader("A. Energi Ikat")

    st.markdown("""
    ### 1. Massa defek

    Hasil pengukuran menunjukkan bahwa massa atom terikat lebih kecil
    daripada jumlah massa partikel penyusunnya ketika terpisah.
    Selisih massa tersebut disebut massa defek (Î”m). Massa yang berkurang
    berkaitan dengan energi ikat yang menjaga nukleon tetap terikat
    di dalam inti.
    """)
    st.latex(
        r"\Delta m=Zm_H+(A-Z)m_n-m_{\mathrm{atom}}"
    )
    st.write("""
    Dalam bentuk ini, mH adalah massa atom hidrogen, mn adalah massa
    neutron, dan massa atom yang digunakan adalah massa atom netral.
    Bentuk tersebut memanfaatkan pembatalan kontribusi elektron.
    Jika menggunakan massa inti, gunakan massa proton dan massa inti,
    bukan mencampur massa atom dengan massa inti.
    """)

    st.markdown("### 2. Energi ikat")
    st.write("""
    Menurut hubungan massa-energi Einstein, massa defek setara dengan
    energi ikat inti. Energi ini diperlukan untuk memisahkan seluruh
    nukleon dari inti menjadi partikel bebas.
    """)
    st.latex(r"B=\Delta m c^2")
    st.latex(r"B(\mathrm{MeV})=\Delta m(\mathrm{u})\times931.5")
    st.markdown("""
    **Keterangan:**
    - B = energi ikat inti.
    - Î”m = massa defek.
    - c = cepat rambat cahaya.
    - 931,5 MeV/u = faktor konversi satuan massa atom ke energi.
    """)

    st.markdown("### 3. Kalkulator massa defek dan energi ikat")
    st.write("Masukkan data massa atom netral dan komposisi nuklidanya.")
    c1,c2=st.columns(2)
    with c1:
        Z=st.number_input("Jumlah proton (Z)",min_value=1,max_value=118,value=6,key="ikat_Z")
        mH=st.number_input("Massa atom hidrogen (u)",value=1.007825,format="%.6f",key="ikat_mH")
    with c2:
        N=st.number_input("Jumlah neutron (N)",min_value=0,max_value=300,value=6,key="ikat_N")
        mn=st.number_input("Massa neutron (u)",value=1.008665,format="%.6f",key="ikat_mn")
    matom=st.number_input("Massa atom netral (u)",min_value=0.000001,value=12.000000,format="%.6f",key="ikat_ma")
    if st.button("Hitung energi ikat",key="hitung_ikat"):
        A=int(Z+N)
        dm=Z*mH+N*mn-matom
        if dm>0:
            B=dm*931.5
            st.success(f"Nomor massa A = {A}")
            q1,q2,q3=st.columns(3)
            q1.metric("Massa defek",f"{dm:.6f} u")
            q2.metric("Energi ikat",f"{B:.3f} MeV")
            q3.metric("Energi per nukleon",f"{B/A:.3f} MeV")
            st.write("Energi ikat per nukleon dihitung dengan B/A.")
        else:
            st.error("Massa defek tidak positif. Periksa kembali data massa yang dimasukkan.")

    st.markdown("### 4. Energi ikat per nukleon")
    st.write("""
    Energi ikat per nukleon diperoleh dengan membagi energi ikat total
    dengan nomor massa A. Besaran ini digunakan untuk membandingkan
    kuatnya ikatan rata-rata tiap nukleon pada berbagai inti.
    """)
    st.latex(r"\frac{B}{A}")
    st.pyplot(binding_curve())
    st.caption("Kurva skematis: energi ikat per nukleon mencapai daerah maksimum di sekitar inti besi, kemudian menurun pada inti berat.")

    st.markdown("### 5. Hubungan energi ikat dengan fisi dan fusi")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("""
        <div class="kartu">
        <h3>Fisi</h3>
        Inti berat membelah menjadi inti-inti yang lebih ringan.
        Produk fisi dapat memiliki energi ikat per nukleon lebih besar,
        sehingga selisih massa dapat dilepaskan sebagai energi.
        </div>
        """,unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kartu">
        <h3>Fusi</h3>
        Dua inti ringan bergabung membentuk inti lebih berat.
        Jika produk memiliki energi ikat per nukleon lebih besar,
        proses penggabungan dapat melepaskan energi.
        </div>
        """,unsafe_allow_html=True)

# ============================================================
# BAB II B. GAYA INTI
# ============================================================
elif menu == "II. Gaya Inti":
    st.header("II. ENERGI IKAT DAN GAYA INTI")
    st.subheader("B. Gaya Inti")

    st.markdown("""
    ### 1. Mengapa gaya inti diperlukan?

    Proton memiliki muatan positif sehingga proton-proton di dalam inti
    saling tolak melalui gaya Coulomb. Namun, inti tetap dapat bertahan
    sebagai satu kesatuan. Hal ini menunjukkan adanya interaksi tarik
    yang sangat kuat pada jarak inti, yaitu gaya inti atau gaya nuklir.
    Dalam buku ajar, gaya ini juga disebut gaya kuat.
    """)

    st.markdown("### 2. Sifat-sifat gaya inti")
    st.markdown("""
    **a. Sangat kuat pada jarak inti.** Gaya inti mampu mengatasi tolakan
    listrik antarproton sehingga nukleon dapat terikat.

    **b. Jangkauannya pendek.** Pengaruhnya dominan pada jarak sekitar
    ukuran inti, kurang lebih 1 fm. Pada jarak yang lebih besar,
    pengaruh gaya inti melemah dengan cepat.

    **c. Sifat kejenuhan.** Setiap nukleon terutama
    berinteraksi dengan nukleon tetangga terdekat. Hal ini membantu
    menjelaskan mengapa rapat inti dan energi ikat per nukleon
    relatif tidak banyak berubah pada banyak inti.

    **d. Hampir tidak bergantung pada jenis nukleon.** Dalam gambaran
    sederhana pada buku, interaksi inti antara pasangan nukleon
    tidak ditentukan semata-mata oleh apakah partikel itu proton
    atau neutron.
    """)
    st.pyplot(force_range_plot())
    st.caption("Grafik potensial gaya inti bersifat skematis untuk menunjukkan jangkauan pendek, bukan kurva eksperimen.")

    st.markdown("### 3. Model gaya tukar Yukawa")
    st.write("""
    Hideki Yukawa mengusulkan model gaya tukar.
    Dalam gambaran ini, proton dan neutron berinteraksi melalui
    pertukaran partikel perantara. Pertukaran tersebut menghasilkan
    interaksi tarik yang mengikat nukleon.
    """)
    fig,ax=plt.subplots(figsize=(7,3))
    fig.patch.set_facecolor("#10263D"); ax.set_facecolor("#10263D")
    ax.add_patch(Circle((1,1.3),.28,color="#FF6262"))
    ax.add_patch(Circle((5,1.3),.28,color="#45A7F5"))
    ax.text(1,1.3,"p",ha="center",va="center",color="white",weight="bold")
    ax.text(5,1.3,"n",ha="center",va="center",color="white",weight="bold")
    ax.annotate("",xy=(4.5,1.3),xytext=(1.5,1.3),
                arrowprops=dict(arrowstyle="<->",color="#F5A623",lw=2))
    ax.text(3,1.65,"Partikel perantara",ha="center",color="#F5A623")
    ax.text(3,.65,"Interaksi melalui pertukaran partikel",ha="center",color="white")
    ax.set_xlim(0,6); ax.set_ylim(.3,2); ax.axis("off")
    st.pyplot(fig)
    st.caption("Ilustrasi konseptual model pertukaran Yukawa.")

    st.markdown("### 4. Jangkauan dan asas ketidakpastian")
    st.write("""
    Buku menghubungkan jangkauan partikel perantara dengan asas
    ketidakpastian Heisenberg. Secara pendekatan, energi massa partikel
    perantara dan jangkauan memenuhi:
    """)
    st.latex(r"\Delta E\,\Delta t \sim \hbar")
    st.latex(r"x \sim c\Delta t")
    st.latex(r"mc^2 \sim \frac{\hbar c}{x}")
    st.write("""
    Dengan jangkauan sekitar 10â»Â¹âµ m, buku memperkirakan energi diam
    partikel perantara berada pada orde sekitar 200 MeV. Partikel
    perantara dalam proses ini disebut partikel virtual; meson dapat
    muncul sebagai partikel nyata dalam kondisi tumbukan berenergi tinggi.
    """)

# ============================================================
# BAB III A. MODEL TETES CAIRAN
# ============================================================
elif menu == "III. Model Tetes Cairan":
    st.header("III. MODEL INTI")
    st.subheader("A. Model Tetes Cairan")

    st.markdown("""
    ### 1. Gagasan dasar model tetes cairan

    Model tetes cairan dikembangkan untuk menjelaskan beberapa sifat
    inti berdasarkan kemiripannya dengan tetesan cairan. C. V. WeizsÃ¤cker
    menghubungkan ukuran, massa, dan energi ikat inti dengan sifat
    tetesan cairan.

    Dalam analogi ini, rapat cairan relatif tetap, ukuran tetesan
    bergantung pada jumlah partikel, dan energi yang diperlukan untuk
    memisahkan partikel berkaitan dengan energi ikat. Nukleon dipandang
    secara kolektif, bukan dihitung satu per satu.
    """)
    st.pyplot(liquid_drop_diagram())
    st.caption("Ilustrasi inti sebagai kumpulan nukleon yang berperilaku secara kolektif.")

    st.markdown("### 2. Persamaan semiempiris massa inti")
    st.write("""
    Model tetes cairan menghasilkan persamaan semiempiris. Bentuk
    massa inti dalam buku memuat massa proton dan neutron serta
    beberapa suku koreksi:
    """)
    st.latex(
        r"\begin{aligned}"
        r"m \approx &\,Zm_p+(A-Z)m_n-b_1A"
        r"+b_2A^{2/3}\\"
        r"&+b_3Z^2A^{-1/3}"
        r"+b_4(A-2Z)^2A^{-1}"
        r"+b_5A^{-3/4}"
        r"\end{aligned}"
    )
    st.write("""
    Koefisien ditentukan dari data eksperimen. Nilai yang dicantumkan
    dalam buku adalah bâ‚ = 14,0 MeV, bâ‚‚ = 13,0 MeV,
    bâ‚ƒ = 0,58 MeV, dan bâ‚„ = 19,3 MeV. Suku pasangan bâ‚…
    bergantung pada kombinasi genap atau ganjil nomor massa dan nomor atom.
    """)

    st.markdown("### 3. Makna koreksi dalam model tetes cairan")
    koreksi = [
        ("Koreksi volume", "Energi ikat bertambah sebanding dengan jumlah nukleon A. Suku ini berkaitan dengan nukleon di bagian dalam inti yang dikelilingi nukleon lain."),
        ("Koreksi permukaan", "Nukleon di permukaan memiliki tetangga lebih sedikit sehingga ikatannya lebih lemah. Besarnya berkaitan dengan luas permukaan, yaitu A^(2/3)."),
        ("Koreksi Coulomb", "Proton saling tolak secara listrik. Tolakan ini mengurangi energi ikat dan berkaitan dengan ZÂ²/A^(1/3)."),
        ("Koreksi asimetri", "Perbedaan jumlah neutron dan proton memengaruhi energi inti. Suku koreksinya berkaitan dengan (N-Z)Â²/A atau (A-2Z)Â²/A."),
        ("Koreksi pasangan", "Nukleon cenderung membentuk pasangan. Efeknya bergantung pada apakah proton dan neutron berjumlah genap atau ganjil.")
    ]
    for title,desc in koreksi:
        with st.expander(title):
            st.write(desc)

    st.markdown("### 4. Energi ikat per nukleon")
    st.write("""
    Dengan membagi energi ikat total terhadap A, diperoleh energi ikat
    rata-rata per nukleon. Bentuk ini memperlihatkan kontribusi koreksi
    volume, permukaan, Coulomb, asimetri, dan pasangan. Pendekatan
    model tetes cairan cukup sesuai dengan kecenderungan eksperimen,
    walaupun tidak menjelaskan seluruh sifat inti secara tepat.
    """)
    st.pyplot(binding_curve())

    st.info(
        "Keterbatasan model: nukleon diperlakukan secara kolektif. "
        "Karena itu, model ini tidak secara rinci menjelaskan tingkat "
        "keadaan eksitasi dan momen magnetik inti."
    )

# ============================================================
# BAB III B. MODEL KULIT
# ============================================================
elif menu == "III. Model Kulit":
    st.header("III. MODEL INTI")
    st.subheader("B. Model Kulit")

    st.markdown("""
    ### 1. Gagasan dasar model kulit

    Berbeda dengan model tetes cairan yang memandang nukleon secara
    kolektif, model kulit memperhitungkan perilaku nukleon secara
    individual. Model ini digunakan untuk menjelaskan sifat inti
    yang tidak cukup diterangkan oleh model tetes cairan, misalnya
    tingkat energi keadaan eksitasi dan momen magnetik inti.
    """)

    st.markdown("### 2. Bilangan ajaib")
    st.write("""
    Data eksperimen menunjukkan adanya kestabilan inti yang menonjol
    ketika jumlah proton Z atau jumlah neutron N mencapai bilangan
    tertentu. Bilangan tersebut dikenal sebagai bilangan ajaib:
    """)
    magic=[2,8,20,28,50,82,126]
    cols=st.columns(len(magic))
    for col,val in zip(cols,magic):
        col.metric("Bilangan",val)
    st.write("""
    Pada bilangan ajaib, inti cenderung lebih stabil dan terdapat
    jumlah isotop stabil yang relatif banyak. Energi untuk melepaskan
    nukleon juga dapat lebih besar dibandingkan inti di luar bilangan
    ajaib.
    """)

    st.markdown("### 3. Tingkat energi nukleon")
    st.write("""
    Dalam pendekatan model kulit, nukleon bergerak di dalam potensial
    rata-rata yang dibentuk oleh nukleon lainnya. Buku memperkenalkan
    pendekatan potensial osilator harmonik:
    """)
    st.latex(r"V(R)=\frac{1}{2}m\omega^2R^2")
    st.write("Tingkat energinya dituliskan:")
    st.latex(r"E=\left(N+\frac{3}{2}\right)\hbar\omega")
    st.write("""
    Dalam persamaan tersebut, N merupakan bilangan kuantum osilator,
    Ï‰ adalah frekuensi sudut, dan Ä§ adalah konstanta Planck tereduksi.
    Keadaan orbital juga dinyatakan dengan bilangan kuantum momentum
    orbital l. Notasi spektroskopik yang digunakan adalah s, p, d, f,
    g, dan seterusnya untuk l = 0, 1, 2, 3, 4, dan seterusnya.
    """)
    st.pyplot(shell_diagram())
    st.caption("Ilustrasi sederhana tingkat energi. Diagram bukan skala energi atau urutan orbit lengkap.")

    st.markdown("### 4. Interaksi spin-orbit")
    st.write("""
    Nukleon memiliki spin s = 1/2. Interaksi antara momentum orbital
    dan spin (interaksi spin-orbit) menyebabkan tingkat dengan l > 0 terpisah
    menjadi dua keadaan, yaitu:
    """)
    st.latex(r"j=l+s \quad \text{atau}\quad j=l-s")
    st.write("""
    Pemisahan tingkat energi ini penting dalam penjelasan bilangan
    ajaib pada model kulit inti.
    """)

    st.markdown("### 5. Prinsip larangan Pauli pada inti")
    st.write("""
    Tidak ada dua nukleon sejenis yang dapat menempati keadaan kuantum
    yang sama secara lengkap. Dalam model kulit, satu orbit dengan
    bilangan momentum total j dapat menampung maksimum:
    """)
    st.latex(r"2j+1")
    st.write("""
    Prinsip ini membantu menjelaskan pengisian kulit proton dan neutron
    serta munculnya bilangan ajaib.
    """)
    st.info(
        "Model kulit dan model tetes cairan saling melengkapi: model "
        "tetes cairan menjelaskan kecenderungan kolektif, sedangkan "
        "model kulit menjelaskan struktur tingkat energi nukleon."
    )

# ============================================================
# SIMULASI
# ============================================================
elif menu == "Simulasi":
    st.header("LABORATORIUM SIMULASI")
    jenis = st.radio(
        "Pilih simulasi",
        [
            "Struktur inti atom",
            "Perbandingan isotop",
            "Energi ikat",
            "Grafik B/A, fisi, fusi, dan gaya inti"
        ],
        horizontal=True
    )

    if jenis == "Struktur inti atom":
        st.write("Atur jumlah proton dan neutron untuk membentuk nuklida.")
        a, b = st.columns(2)
        with a:
            Z = st.slider("Jumlah proton", 1, 20, 6)
        with b:
            N = st.slider("Jumlah neutron", 0, 30, 6)
        A = Z + N
        st.pyplot(nucleus_diagram(Z, N, seed=A + Z))
        c1, c2, c3 = st.columns(3)
        c1.metric("Proton (Z)", Z)
        c2.metric("Neutron (N)", N)
        c3.metric("Nomor massa (A)", A)
        st.latex(r"A=Z+N")
        st.write(
            f"Nuklida yang terbentuk memiliki nomor atom {Z} "
            f"dan nomor massa {A}."
        )
        st.caption(
            "Visualisasi merupakan model skematis, bukan posisi "
            "nukleon yang sebenarnya."
        )

    elif jenis == "Perbandingan isotop":
        st.write("Pilih unsur, kemudian ubah jumlah neutron pada kedua inti.")
        Z = st.selectbox(
            "Nomor atom", [1, 2, 3, 6, 7, 8, 11, 12],
            format_func=lambda z: {
                1: "Hidrogen", 2: "Helium", 3: "Litium", 6: "Karbon",
                7: "Nitrogen", 8: "Oksigen", 11: "Natrium",
                12: "Magnesium"
            }[z]
        )
        n1 = st.slider("Neutron inti pertama", 0, 20, 6)
        n2 = st.slider("Neutron inti kedua", 0, 20, 7)
        x, y = st.columns(2)
        with x:
            st.pyplot(nucleus_diagram(
                Z, n1, seed=1, title=f"Inti pertama: A={Z+n1}"
            ))
            st.metric("Nomor massa pertama", Z + n1)
        with y:
            st.pyplot(nucleus_diagram(
                Z, n2, seed=2, title=f"Inti kedua: A={Z+n2}"
            ))
            st.metric("Nomor massa kedua", Z + n2)
        if n1 != n2:
            st.success("Keduanya merupakan isotop: Z sama, N berbeda.")
        else:
            st.warning(
                "Jumlah neutron sama. Ubah salah satu nilai untuk "
                "membuat dua isotop berbeda."
            )

    elif jenis == "Energi ikat":
        st.write(
            "Kalkulator menggunakan massa atom netral dan massa "
            "atom hidrogen."
        )
        Z = st.number_input("Proton", 1, 118, 6, key="simZ")
        N = st.number_input("Neutron", 0, 300, 6, key="simN")
        mH = st.number_input(
            "Massa atom hidrogen (u)", value=1.007825,
            format="%.6f", key="simH"
        )
        mn = st.number_input(
            "Massa neutron (u)", value=1.008665,
            format="%.6f", key="simn"
        )
        ma = st.number_input(
            "Massa atom netral (u)", value=12.000000,
            format="%.6f", key="sima"
        )
        if st.button("Hitung", key="simhitung"):
            dm = Z * mH + N * mn - ma
            if dm > 0:
                B = dm * 931.5
                st.metric("Massa defek", f"{dm:.6f} u")
                st.metric("Energi ikat", f"{B:.3f} MeV")
                st.metric("Energi per nukleon", f"{B/(Z+N):.3f} MeV")
            else:
                st.error("Periksa kembali data massa.")

    elif jenis == "Grafik B/A, fisi, fusi, dan gaya inti":
        st.subheader("1. Simulasi grafik energi ikat per nukleon dan reaksi")
        st.write(
            "Pilih reaksi untuk melihat energi reaksi Q, kemudian ubah "
            "massa salah satu inti dan amati pergeseran titik pada grafik."
        )
        s1, s2 = st.columns(2)
        with s1:
            nama_reaksi = st.selectbox(
                "Pilih reaksi", list(REAKSI.keys()), key="sim_reaksi_baru"
            )
        with s2:
            nama_inti = list(DATA_INTI.keys())
            inti_diubah = st.selectbox(
                "Inti yang massanya diubah", nama_inti,
                index=nama_inti.index("Fe-56"), key="sim_inti_ubah_baru"
            )
        dm = st.slider(
            "Perubahan massa atom (u)", -0.030, 0.030, 0.0, 0.0005,
            format="%.4f", key="sim_dm_baru"
        )
        fig_ba, Q, ba_terpilih = grafik_ba_reaksi(
            nama_reaksi, inti_diubah, dm
        )
        st.pyplot(fig_ba)
        q1, q2 = st.columns(2)
        q1.metric("Energi reaksi (Q)", f"{Q:.1f} MeV")
        q2.metric(f"B/A {inti_diubah}", f"{ba_terpilih:.3f} MeV")
        st.caption(
            "Lingkaran oranye = inti pereaksi, lingkaran putih = inti hasil. "
            "Q positif berarti reaksi melepas energi."
        )
        pereaksi_terpilih, hasil_terpilih = REAKSI[nama_reaksi]
        if inti_diubah not in pereaksi_terpilih + hasil_terpilih:
            st.info(
                "Inti yang dipilih tidak terlibat dalam reaksi ini. "
                "Titik B/A bergeser, tetapi nilai Q tidak berubah."
            )

        st.divider()
        st.subheader("2. Simulasi bukti jangkauan pendek dari data")
        st.write(
            "Jika gaya inti berjangkauan panjang, energi ikat total akan "
            "tumbuh sebanding A² dan energi per nukleon terus naik. Data "
            "massa inti yang digunakan di sini dibandingkan dengan "
            "kecenderungan E sebanding A dan E/A yang hampir datar."
        )
        p1, p2 = st.columns(2)
        with p1:
            a_pendek = st.slider(
                "a: B/A jangkauan pendek (MeV)", 6.0, 10.0, 8.0, 0.1,
                key="sim_a_baru"
            )
            k_panjang = st.slider(
                "k: koefisien jangkauan panjang (MeV)", 0.010, 0.100,
                0.032, 0.001, format="%.3f", key="sim_k_baru"
            )
        with p2:
            r0_sim = st.slider(
                "r₀ (fm)", 1.0, 1.5, 1.2, 0.01, key="sim_r0_baru"
            )
            A_kursor = st.slider(
                "Nomor massa A (kursor)", 2, 250, 100, 1,
                key="sim_Akursor_baru"
            )
        fig_jp, rho = grafik_jangkauan_pendek(
            a_pendek, k_panjang, r0_sim, A_kursor
        )
        st.pyplot(fig_jp)
        st.write(
            f"Kerapatan inti = **{rho:.3f} nukleon/fm³**, sama untuk "
            f"semua A. Pada A = {A_kursor}, gaya jangkauan pendek "
            f"memberi E/A = **{a_pendek:.2f} MeV**, sedangkan gaya "
            f"jangkauan panjang memberi **{k_panjang*A_kursor:.2f} MeV**."
        )

        st.divider()
        st.subheader("3. Simulasi jangkauan gaya dari massa partikel perantara")
        y1, y2 = st.columns(2)
        with y1:
            E_mc2 = st.slider(
                "Energi diam partikel, mc² (MeV)", 10, 1000, 200, 5,
                key="sim_E_baru"
            )
        with y2:
            d_nukleon = st.slider(
                "Jarak kedua nukleon (fm)", 0.3, 4.0, 1.0, 0.05,
                key="sim_dnuk_baru"
            )
        fig_y, x_jangkau, dt_hidup, sampai = grafik_yukawa(
            E_mc2, d_nukleon
        )
        st.pyplot(fig_y)
        m1, m2 = st.columns(2)
        m1.metric("Jangkauan x = ħc/mc²", f"{x_jangkau:.2f} fm")
        m2.metric(
            "Waktu hidup Δt = ħ/mc²",
            f"{dt_hidup*1e24:.2f} × 10⁻²⁴ s"
        )
        if sampai:
            st.success("Partikel perantara mencapai nukleon lain: gaya inti bekerja.")
        else:
            st.warning(
                "Jarak lebih besar dari jangkauan: gaya inti tidak bekerja."
            )

# ============================================================
# LATIHAN SOAL (DIPERTAHANKAN)
# ============================================================
elif menu == "Latihan Soal":
    st.header("LATIHAN SOAL")
    soal=[
        ("Partikel penyusun inti atom adalah ...",
         ["Elektron dan proton","Proton dan neutron","Elektron dan neutron","Proton dan foton"],
         "Proton dan neutron",
         "Inti atom tersusun atas proton dan neutron yang disebut nukleon."),
        ("Nomor atom Z menunjukkan jumlah ...",
         ["Neutron","Nukleon","Proton","Proton dan neutron"],
         "Proton",
         "Nomor atom menunjukkan jumlah proton."),
        ("Jika A=23 dan Z=11, jumlah neutron adalah ...",
         ["11","12","23","34"],"12",
         "N=A-Z=23-11=12."),
        ("Nuklida dengan Z sama tetapi A berbeda disebut ...",
         ["Isobar","Isoton","Isotop","Isomer"],"Isotop",
         "Isotop memiliki proton sama tetapi neutron berbeda."),
        ("Satuan massa atom didefinisikan berdasarkan ...",
         ["Hidrogen-1","Karbon-12","Oksigen-16","Uranium-235"],
         "Karbon-12",
         "Satu u didefinisikan sebagai 1/12 massa atom karbon-12."),
        ("Rumus jari-jari inti adalah ...",
         ["R=R0 A","R=R0 A^(1/3)","R=R0/A","R=R0 A^2"],
         "R=R0 A^(1/3)",
         "Jari-jari inti sebanding dengan pangkat sepertiga nomor massa."),
        ("Energi ikat berkaitan dengan ...",
         ["Massa defek","Muatan elektron","Jari-jari atom","Nomor periode"],
         "Massa defek",
         "Energi ikat berasal dari kesetaraan massa-energi massa defek."),
        ("Gaya inti memiliki jangkauan ...",
         ["Sangat panjang","Sekitar ukuran inti","Tak terbatas","Seukuran atom"],
         "Sekitar ukuran inti",
         "Gaya inti dominan pada jarak sekitar femtometer."),
        ("Bilangan ajaib dalam model kulit mencakup ...",
         ["3, 5, 7","2, 8, 20, 28","1, 4, 9","10, 15, 25"],
         "2, 8, 20, 28",
         "Deret bilangan ajaib yang dibahas buku meliputi 2, 8, 20, 28, 50, 82, dan 126."),
        ("Model yang memandang nukleon secara kolektif adalah ...",
         ["Model kulit","Model tetes cairan","Model Bohr","Model Thomson"],
         "Model tetes cairan",
         "Model tetes cairan memperlakukan nukleon secara kolektif.")
    ]

    with st.form("form_latihan"):
        jawaban=[]
        for i,(pertanyaan,opsi,kunci,pembahasan) in enumerate(soal):
            st.markdown(f"**{i+1}. {pertanyaan}**")
            jawaban.append(st.radio(
                "Pilih jawaban:",opsi,key=f"latihan_{i}",index=None
            ))
        kirim=st.form_submit_button("Periksa Jawaban")

    if kirim:
        if any(j is None for j in jawaban):
            st.warning("Jawab semua pertanyaan terlebih dahulu.")
        else:
            benar=sum(jawaban[i]==soal[i][2] for i in range(len(soal)))
            nilai=benar/len(soal)*100
            st.session_state.nilai=nilai
            st.session_state.benar=benar
            st.divider()
            st.subheader("Hasil Latihan")
            c1,c2=st.columns(2)
            c1.metric("Jawaban benar",f"{benar}/{len(soal)}")
            c2.metric("Nilai",f"{nilai:.0f}")
            if nilai>=75:
                st.success("Selamat, kamu mencapai ketuntasan.")
            else:
                st.warning("Pelajari kembali materi dan coba lagi.")
            with st.expander("Lihat kunci dan pembahasan"):
                for i,item in enumerate(soal):
                    st.markdown(f"**{i+1}. {item[0]}**")
                    st.write(f"Jawaban kamu: {jawaban[i]}")
                    st.write(f"Jawaban benar: {item[2]}")
                    st.caption(item[3])
                    st.divider()

# ============================================================
# EVALUASI (DIPERTAHANKAN)
# ============================================================
elif menu == "Evaluasi":
    st.header("EVALUASI PEMBELAJARAN")
    st.write("Evaluasi menampilkan hasil latihan yang telah dikerjakan.")
    if "nilai" in st.session_state:
        c1,c2=st.columns(2)
        c1.metric("Nilai terakhir",f"{st.session_state.nilai:.0f}")
        c2.metric("Jawaban benar",st.session_state.benar)
        if st.session_state.nilai>=75:
            st.success("Kriteria ketuntasan tercapai.")
        else:
            st.warning("Kriteria ketuntasan belum tercapai. Ulangi latihan setelah mempelajari materi.")
    else:
        st.info("Belum ada hasil. Silakan kerjakan latihan soal terlebih dahulu.")

# --------------------------- FOOTER --------------------------
st.divider()
st.markdown("""
<div style="text-align:center;color:#A9C4D8;padding:12px">
<b>NUCLEAR LAB</b><br>
Laboratorium Virtual Fisika Inti<br>
Media Pembelajaran Fisika Inti
</div>
""",unsafe_allow_html=True)