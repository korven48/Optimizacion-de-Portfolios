#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/quire.hpp>
#include <universal/number/posit/fdp.hpp>
#include <cmath>
#include <sstream>
#include <vector>

namespace py = pybind11;
using namespace sw::universal;

// Wrapper de plantilla para diferentes configuraciones de posit
template<size_t nbits, size_t es>
class PositWrapper {
private:
    posit<nbits, es> value;

public:
    // Constructores
    PositWrapper() : value(0) {}
    PositWrapper(double d) : value(d) {}
    PositWrapper(int i) : value(i) {}
    PositWrapper(const posit<nbits, es>& p) : value(p) {}
    
    // Conversión a double
    double to_double() const {
        return double(value);
    }
    
    // Operadores aritméticos
    PositWrapper operator+(const PositWrapper& other) const {
        return PositWrapper(value + other.value);
    }
    
    PositWrapper operator-(const PositWrapper& other) const {
        return PositWrapper(value - other.value);
    }
    
    PositWrapper operator*(const PositWrapper& other) const {
        return PositWrapper(value * other.value);
    }
    
    PositWrapper operator/(const PositWrapper& other) const {
        return PositWrapper(value / other.value);
    }
    
    PositWrapper operator-() const {
        return PositWrapper(-value);
    }
    
    PositWrapper operator+() const {
        return *this;
    }
    
    // Operadores de comparación
    bool operator==(const PositWrapper& other) const {
        return value == other.value;
    }
    
    bool operator!=(const PositWrapper& other) const {
        return value != other.value;
    }
    
    bool operator<(const PositWrapper& other) const {
        return value < other.value;
    }
    
    bool operator>(const PositWrapper& other) const {
        return value > other.value;
    }
    
    bool operator<=(const PositWrapper& other) const {
        return value <= other.value;
    }
    
    bool operator>=(const PositWrapper& other) const {
        return value >= other.value;
    }
    
    // Representación en cadena
    std::string repr() const {
        std::ostringstream oss;
        oss << "Posit" << nbits << "<" << es << ">(" << double(value) << ")";
        return oss.str();
    }
    
    std::string str() const {
        std::ostringstream oss;
        oss << double(value);
        return oss.str();
    }
    
    // Funciones matemáticas
    PositWrapper sqrt_() const {
        return PositWrapper(sw::universal::sqrt(value));
    }
    
    PositWrapper exp_() const {
        return PositWrapper(sw::universal::exp(value));
    }
    
    PositWrapper log_() const {
        return PositWrapper(sw::universal::log(value));
    }
    
    PositWrapper abs_() const {
        return PositWrapper(sw::universal::abs(value));
    }
    
    PositWrapper pow_(const PositWrapper& exp) const {
        return PositWrapper(sw::universal::pow(value, exp.value));
    }

    // Producto escalar usando el quire
    // Capacity=20 soporta vectores de hasta 2^20 = 1M elementos sin overflow.
    PositWrapper dot_product_quire(const std::vector<PositWrapper>& v1,
                                   const std::vector<PositWrapper>& v2) const {
        constexpr unsigned capacity = 20;
        sw::universal::quire<nbits, es, capacity> q(0);
        for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
            q += sw::universal::quire_mul(v1[i].value, v2[i].value);
        }
        posit<nbits, es> result;
        sw::universal::convert(q.to_value(), result);  // único redondeo
        return PositWrapper(result);
    }
};

// Definición del módulo
PYBIND11_MODULE(posit, m) {
    m.doc() = "Wrapper aritmético de Posit para optimización de portafolios";
    
    // Posit8<2>
    py::class_<PositWrapper<8, 2>>(m, "Posit8")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<8, 2>::to_double)
        .def("__repr__", &PositWrapper<8, 2>::repr)
        .def("__str__", &PositWrapper<8, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<8, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<8, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<8, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<8, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<8, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<8, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<8, 2>::dot_product_quire, "Producto punto exacto con quire");

    // Posit12<2>
    py::class_<PositWrapper<12, 2>>(m, "Posit12")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<12, 2>::to_double)
        .def("__repr__", &PositWrapper<12, 2>::repr)
        .def("__str__", &PositWrapper<12, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<12, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<12, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<12, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<12, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<12, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<12, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<12, 2>::dot_product_quire, "Producto punto exacto con quire");

    // Posit16<2>
    py::class_<PositWrapper<16, 2>>(m, "Posit16")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<16, 2>::to_double)
        .def("__repr__", &PositWrapper<16, 2>::repr)
        .def("__str__", &PositWrapper<16, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<16, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<16, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<16, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<16, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<16, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<16, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<16, 2>::dot_product_quire, "Producto punto exacto con quire");

    // Posit20<2>
    py::class_<PositWrapper<20, 2>>(m, "Posit20")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<20, 2>::to_double)
        .def("__repr__", &PositWrapper<20, 2>::repr)
        .def("__str__", &PositWrapper<20, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<20, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<20, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<20, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<20, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<20, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<20, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<20, 2>::dot_product_quire, "Producto punto exacto con quire");

    // Posit24<2>
    py::class_<PositWrapper<24, 2>>(m, "Posit24")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<24, 2>::to_double)
        .def("__repr__", &PositWrapper<24, 2>::repr)
        .def("__str__", &PositWrapper<24, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<24, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<24, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<24, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<24, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<24, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<24, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<24, 2>::dot_product_quire, "Producto punto exacto con quire");

    // Posit32<2> - Configuración estándar
    py::class_<PositWrapper<32, 2>>(m, "Posit32")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<32, 2>::to_double)
        .def("__repr__", &PositWrapper<32, 2>::repr)
        .def("__str__", &PositWrapper<32, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<32, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<32, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<32, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<32, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<32, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<32, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<32, 2>::dot_product_quire, "Producto punto exacto con quire");
    
    // Posit64<2> - Configuración de alta precisión
    py::class_<PositWrapper<64, 2>>(m, "Posit64")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<64, 2>::to_double)
        .def("__repr__", &PositWrapper<64, 2>::repr)
        .def("__str__", &PositWrapper<64, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<64, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<64, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<64, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<64, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<64, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<64, 2>::pow_)
        .def("dot_product_quire", &PositWrapper<64, 2>::dot_product_quire, "Producto punto exacto con quire");
    
    // Funciones a nivel de módulo
    m.def("sqrt", [](const PositWrapper<8, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit8)");
    m.def("sqrt", [](const PositWrapper<12, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit12)");
    m.def("sqrt", [](const PositWrapper<16, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit16)");
    m.def("sqrt", [](const PositWrapper<20, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit20)");
    m.def("sqrt", [](const PositWrapper<24, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit24)");
    m.def("sqrt", [](const PositWrapper<32, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit32)");
    m.def("sqrt", [](const PositWrapper<64, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit64)");
}
