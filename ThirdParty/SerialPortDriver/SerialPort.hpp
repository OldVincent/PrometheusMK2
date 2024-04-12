#pragma once

#include <boost/asio.hpp>
#include <vector>
#include <string>

namespace RoboPioneers::Modules::SerialPortDriver
{
	/**
	 * @brief 串口类
	 * @author Vincent
	 * @details
	 *  ~ 一个该类对象对应一个操作系统的串口文件对象。
	 *  ~ 该类封装了对串口的基本设置和读写操作。
	 *  ~ 仅少数允许静默处理的异常消息将通过std::clog输出。
	 */
	class SerialPort
	{
	private:
		/**
		 * @brief 设备名称
		 * @details 设备在系统中的显示名。
		 */
		std::string DeviceName;

	protected:
		///
		boost::asio::io_context Context;
		/// 串口对象
		boost::asio::serial_port Port;
	public:
		/**
		 * @brief 缓存区大小，为字节的个数，单位为1
		 * @details
		 *  ~ 修改该值后将对所有使用默认缓存区大小设定的方法立即生效。
		 *  ~ 如果缓存区较小，则一段消息需要读取多次。
		 */
		std::size_t BufferSize {1024};

		//==============================
		// 构造与析构函数部分
		//==============================

		/// 默认构造函数，不绑定设备，需要在后续打开时给出设备文件名称
		SerialPort();

		/**
		 * @brief 构造函数，将尝试自动打开端口
		 * @param device_name 设备名称
		 * @throw boost::system::system_error 当设备打开失败
		 * @details 当自动打开设备失败时，将向std::clog流输出该消息。
		 */
		explicit SerialPort(std::string device_name) noexcept;

		/// 析构函数，若设备未关闭则将关闭设备
		virtual ~SerialPort() noexcept;

		//==============================
		// 基本控制部分
		//==============================

		/**
		 * @brief 打开端口，通常将在系统级别独占该端口设备
		 * @throw boost::system::system_error 当打开指定设备失败
		 * @throw std::logic_error 当当前对象已经打开了设备
		 * @details
		 *  ~ 设备正常时，该方法会在构造时被自动调用。
		 */
		void Open();

		/**
		 * @brief 打开端口，通常将在系统级别独占该端口设备
		 * @param device_name 串口设备在系统中对应的文件名称
		 * @throw boost::system::system_error 当打开指定设备失败
		 * @throw std::logic_error 当当前对象已经打开了设备
		 * @details
		 *  ~ 设备正常时，该方法会在构造时被自动调用。
		 */
		void Open(const std::string& device_name);

		/**
		 * @brief 关闭端口
		 * @throw std::logic_error 当设备未打开
		 * @details
		 *  ~ 若析构时设备未被关闭，则该方法会被自动调用。
		 */
		void Close();

		/**
		 * @brief 查询设备是否已经打开
		 * @return 打开则返回true，未打开则返回false
		 */
		[[nodiscard]] bool IsOpened() const noexcept
		{
			return Port.is_open();
		}

		//==============================
		// 基本设置部分
		// 当设备处于打开状态时这些方法才能生效
		//==============================

		/**
		 * @brief 设置波特率
		 * @param value 波特率的值
		 * @pre 设备已经被成功打开
		 * @throw boost::system::system_error 当设置失败
		 * @throw std::logic_error 当设备未打开
		 */
		void SetBaudRate(unsigned int value = 115200);


		/// 流控制枚举类型别名
		using flow_control = boost::asio::serial_port::flow_control;

		/**
		 * @brief 设置流控制类型
		 * @param type 控制类型
		 * @pre 设备已经被成功打开
		 * @throw boost::system::system_error 当设置失败
		 * @throw std::logic_error 当设备未打开
		 */
		void SetFlowControlType(flow_control::type type = flow_control::none);

		/// 奇偶校验枚举类型别名
		using parity = boost::asio::serial_port::parity;

		/**
		 * @brief 设置奇偶校验类型
		 * @param type 校验类型
		 * @pre 设备已经被成功打开
		 * @throw boost::system::system_error 当设置失败
		 * @throw std::logic_error 当设备未打开
		 */
		void SetParityType(parity::type type = parity::none);

		/// 停止位枚举类型别名
		using stop_bits = boost::asio::serial_port::stop_bits;

		/**
		 * @brief 设置停止位
		 * @param type 停止位类型
		 * @pre 设备已经被成功打开
		 * @throw boost::system::system_error 当设置失败
		 * @throw std::logic_error 当设备未打开
		 */
		 void SetStopBitsType(stop_bits::type type = stop_bits::one);

		 /**
		  * @brief 设置字符大小
		  * @param size 大小，单位为bit，一般为7或8
		  * @pre 设备已经被成功打开
		  * @throw boost::system::system_error 当设置失败
		  * @throw std::logic_error 当设备未打开
		  */
		 void SetCharacterSize(unsigned int size = 8);

		//==============================
		// 读写操作部分
		//==============================

		/**
		 * @brief 写入字节向量数据
		 * @param data 数据
		 * @throw std::logic_error 当设备未打开
		 * @throw boost::system::system_error 当写入失败
		 * @details
		 *  ~ 在所有的数据都写入完成前将阻塞调用线程。
		 */
		void Write(const std::vector<unsigned char>& data);

		/**
		 * @brief 写入字符串数据
		 * @param text 文本
		 * @throw std::logic_error 当设备未打开
		 * @throw boost::system::system_error 当写入失败
		 */
		void Write(const std::string& text);

		/**
		 * @brief 写入数据
		 * @param start_address 起始地址
		 * @param length 写入的数据的长度，即写入的字节的个数，单位为1
		 * @throw std::logic_error 当设备未打开
		 * @throw boost::system::system_error 当写入失败
		 */
		void Write(const void* start_address, std::size_t length);

		/**
		 * @brief 读取数据
		 * @return 字节向量
		 * @throw std::logic_error 当设备未打开
		 * @throw boost::system::system_error 当写入失败
		 * @details
		 *  ~ 该方法会读取缓冲区中目前存在的不大于1024个字节，随后立即返回。
		 */
		std::vector<unsigned char> Read();

		/**
		 * @brief 读取指定个数的字节
		 * @param length 读取的字节的长度，返回的向量一定具有该长度。
		 * @return 字节向量
		 * @throw std::logic_error 当设备未打开
		 * @throw boost::system::system_error 当写入失败
		 * @details
		 *  ~ 该方法会阻塞直到读取了指定数量的字节。
		 */
		std::vector<unsigned char> Read(std::size_t length);

		/**
		 * @brief 读取字符串
		 * @return 字符串
		 */
		std::string ReadString();

		/**
		 * @brief 读取字符串
		 * @param length 读取的字符串内字符数量，读取到该数量之前不会返回
		 * @return 读取的字符串
		 */
		std::string ReadString(std::size_t length);
	};
}