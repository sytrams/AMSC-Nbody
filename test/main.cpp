#include <gtest/gtest.h>
#include <cmath>
#include <utility>

import vector;

// Test 1: Costruttore Vector<2> e accesso x(), y()
TEST(VectorTest, Constructor2D) {
    Vector<2> a(1.0, 2.0);
    EXPECT_DOUBLE_EQ(a.x(), 1.0);
    EXPECT_DOUBLE_EQ(a.y(), 2.0);
}

// Test 2: Calcolo della norma in Vector<3>
TEST(VectorTest, NormCalculation) {
    Vector<3> b(3.0, 4.0, 12.0);
    // 3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 169. sqrt(169) = 13
    EXPECT_DOUBLE_EQ(b.norm(), 13.0);
}

// Test 3: Operatore [] e norma in Vector<4>
TEST(VectorTest, ElementAccessAndNorm4D) {
    Vector<4> v4;
    v4[0] = 1.0; v4[1] = 2.0; v4[2] = 3.0; v4[3] = 4.0;
    
    double expected_norm = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    
    EXPECT_DOUBLE_EQ(v4[0], 1.0);
    EXPECT_DOUBLE_EQ(v4[1], 2.0);
    EXPECT_DOUBLE_EQ(v4[2], 3.0);
    EXPECT_DOUBLE_EQ(v4[3], 4.0);
    EXPECT_NEAR(v4.norm(), expected_norm, 1e-9);
}

// Test 4: Somma tra vettori
TEST(VectorTest, Addition) {
    Vector<2> a(1.0, 2.0);
    Vector<2> b(2.0, 3.0);
    Vector<2> sum = a + b;
    
    EXPECT_DOUBLE_EQ(sum.x(), 3.0);
    EXPECT_DOUBLE_EQ(sum.y(), 5.0);
}

// Test 5: Moltiplicazione per scalare
TEST(VectorTest, ScalarMultiplication) {
    Vector<3> b(3.0, 4.0, 12.0);
    Vector<3> d = b * 0.5;
    
    EXPECT_DOUBLE_EQ(d.x(), 1.5);
    EXPECT_DOUBLE_EQ(d.y(), 2.0);
    EXPECT_DOUBLE_EQ(d.z(), 6.0);
}

// Test 6: Copy Constructor
TEST(VectorTest, CopyConstructor) {
    Vector<2> a(1.0, 2.0);
    Vector<2> copyA = a;
    
    EXPECT_DOUBLE_EQ(copyA.x(), a.x());
    EXPECT_DOUBLE_EQ(copyA.y(), a.y());
}

// Test 7: Move Constructor
TEST(VectorTest, MoveConstructor) {
    Vector<2> a(1.0, 2.0);
    Vector<2> tmp = a; 
    Vector<2> moved = std::move(tmp);
    
    EXPECT_DOUBLE_EQ(moved.x(), 1.0);
    EXPECT_DOUBLE_EQ(moved.y(), 2.0);
}

// Test 8: Operatore +=
TEST(VectorTest, CompoundAssignment) {
    Vector<2> a(1.0, 2.0);
    a += Vector<2>(0.5, 0.5);
    
    EXPECT_DOUBLE_EQ(a.x(), 1.5);
    EXPECT_DOUBLE_EQ(a.y(), 2.5);
}

// Non serve il main se linki gtest_main, 
// altrimenti usa questo:
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
