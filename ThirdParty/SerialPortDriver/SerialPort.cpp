#include "SerialPort.hpp"

#include <iostream>
#include <stdexcept>

namespace RoboPioneers::Modules::SerialPortDriver
{
	/// 构造函数
	SerialPort::SerialPort(std::string device_name) noexcept :
		Context(), Port(Context), DeviceName(std::move(device_name))
	{
		try {
			Port.open(DeviceName);
		}
		catch (boost::system::system_error& error)
		{
			std::clog << "Port::Port Failed to Open Device " << DeviceName << " Automatically." << std::endl;
		}
	}

	/// 默认构造函数
	SerialPort::SerialPort(): Context(), Port(Context)
	{}

	/// 析构函数
	SerialPort::~SerialPort() noexcept
	{
		if (IsOpened())
		{
			Port.close();
		}
	}


	/// 打开端口
	void SerialPort::Open()
	{
		if (!DeviceName.empty())
		{
			if (!IsOpened())
			{
				Port.open(DeviceName);
			}
		}
		else
		{
			throw std::logic_error("Port::Open Target Device Name Is Not Given.");
		}
	}

	/// 打开端口
	void SerialPort::Open(const std::string& device_name)
	{
		if (!Port.is_open())
		{
			Port.open(device_name);
		}
		else
		{
			Port.close();
			DeviceName = device_name;
			Port.open(DeviceName);
		}
	}


	/// 关闭端口
	void SerialPort::Close()
	{
		if (Port.is_open())
		{
			Port.close();
		}
		else
		{
			throw std::logic_error("Port::Close Device is not opened.");
		}
	}

	/// 写入字节向量数据
	void SerialPort::Write(const std::vector<unsigned char>& data)
	{
		Write(static_cast<const void*>(data.data()), data.size());
	}

	/// 写入字符串数据
	void SerialPort::Write(const std::string &text)
	{
		Write(static_cast<const void*>(text.data()), text.size());
	}

	/// 写入数据
	void SerialPort::Write(const void *start_address, std::size_t length)
	{
		if (Port.is_open())
		{
			// 注意此处是write，即等待缓冲区内容全部被写入后才返回；如果是write_some，则会立即返回，返回值标注有多少个字节已写入
			boost::asio::write(Port, boost::asio::buffer(start_address, length));
		}
		else
		{
			throw std::logic_error("Port::Write Device is not opened.");
		}
	}

	/// 读取数据
	std::vector<unsigned char> SerialPort::Read()
	{
		if (Port.is_open())
		{
			std::vector<unsigned char> buffer;
			buffer.resize(BufferSize);
			// 注意此处时read_some，如果是read，则会阻塞直到缓冲区被填满
			std::size_t length = Port.read_some(boost::asio::buffer(buffer.data(), buffer.size()));
			buffer.resize(length);

			return buffer;
		}
		else
		{
			throw std::logic_error("Port::Read Device is not opened.");
		}
	}

	/// 读取指定数量的字节
	std::vector<unsigned char> SerialPort::Read(std::size_t length)
	{
		if (Port.is_open())
		{
			std::vector<unsigned char> buffer;
			buffer.resize(length);
			// 此处会阻塞直到缓冲区被填满
			boost::asio::read(Port, boost::asio::buffer(buffer.data(), buffer.size()));
			return buffer;
		}
		else
		{
			throw std::logic_error("Port::Read Device is not opened.");
		}
	}

	/// 读取字符串
	std::string SerialPort::ReadString()
	{
		if (Port.is_open())
		{
			std::string buffer;
			buffer.resize(BufferSize);
			std::size_t length = Port.read_some(boost::asio::buffer(buffer.data(), buffer.size()));
			buffer.resize(length);
			return buffer;
		}
		else
		{
			throw std::logic_error("Port::Read Device is not opened.");
		}
	}

	/// 读取字符串
	std::string SerialPort::ReadString(std::size_t length)
	{
		if (Port.is_open())
		{
			std::string buffer;
			buffer.resize(length);
			// 此处会阻塞直到缓冲区被填满
			boost::asio::read(Port, boost::asio::buffer(buffer.data(), buffer.size()));
			return buffer;
		}
		else
		{
			throw std::logic_error("Port::Read Device is not opened.");
		}
	}

	/// 设置波特率
	void SerialPort::SetBaudRate(unsigned int value)
	{
		if (Port.is_open())
		{
			Port.set_option(boost::asio::serial_port::baud_rate(value));
		}
		else
		{
			throw std::logic_error("Port::SetBaudRate Device is not opened.");
		}
	}

	/// 设置流控制类型
	void SerialPort::SetFlowControlType(boost::asio::serial_port::flow_control::type type)
	{
		if (Port.is_open())
		{
			Port.set_option(boost::asio::serial_port::flow_control(type));
		}
		else
		{
			throw std::logic_error("Port::SetFlowControlType Device is not opened.");
		}
	}

	/// 设置奇偶校验类型
	void SerialPort::SetParityType(boost::asio::serial_port::parity::type type)
	{
		if (Port.is_open())
		{
			Port.set_option(boost::asio::serial_port::parity(type));
		}
		else
		{
			throw std::logic_error("Port::SetParityType Device is not opened.");
		}
	}

	/// 设置停止位
	void SerialPort::SetStopBitsType(boost::asio::serial_port::stop_bits::type type)
	{
		if (Port.is_open())
		{
			Port.set_option(boost::asio::serial_port::stop_bits(type));
		}
		else
		{
			throw std::logic_error("Port::SetStopBitsType Device is not opened.");
		}
	}

	/// 设置字符大小
	void SerialPort::SetCharacterSize(unsigned int size)
	{
		if (Port.is_open())
		{
			Port.set_option(boost::asio::serial_port::character_size(size));
		}
		else
		{
			throw std::logic_error("Port::SetCharacterSize Device is not opened.");
		}
	}
}